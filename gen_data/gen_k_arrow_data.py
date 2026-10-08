"""
gen_k_arrow_data.py — Generate K-arrow training/val/test data.

每张图按 arrow_id 升序，滑窗 K 个一组（不重叠）；如尾巴 < K 则保留小组。
e.g. 5 arrows, K=2 → groups of (2,2,1).

输出：
  - 新的标注图像：annotated_images_k{K}/{folder}_arrows_<i>-<j>...png
    (在原图上画 N 个蓝色矩形 + 数字 1..N)
  - 新的训练 JSON：data_4_training/{ds}_k{K}_{split}_arrow_training.json

K=1 答案从已有 {ds}_(test/val_)?arrow_training_all.json 取出（保证 schema 一致，
包括 BPMN 的 pool/lane）。Bboxes 从对应 merged JSON 取。

Usage:
  python gen_k_arrow_data.py --dataset cbd --split train --k 2
  python gen_k_arrow_data.py --dataset cbd --split val --k 2
  python gen_k_arrow_data.py --dataset all --split all --k 2,3
"""
import argparse
import json
import os
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2

# 让 prompt 可 import。FLOWQA_ROOT 可在迁移到其他机器后显式覆盖。
PROJECT_ROOT = Path(
    os.environ.get("FLOWQA_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from lora.k_prompt import get_k_triple, get_k_bpmn_triple, get_k_flowgen_triple

DATA_ROOT = Path(
    os.environ.get("FLOWQA_DATA_ROOT", PROJECT_ROOT / "Dataset/data_4_training")
).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# 配置：每个 (dataset, split) → (training_all.json, merged_dir_root, orig_img_dir,
#                                  merged_filename_template, is_bpmn)
# ─────────────────────────────────────────────────────────────────────────────
# 数据集 → prompt 类型映射
# 'cbdfcb'  : get_k_triple (3 字段: source_nodes/end_nodes/condition)
# 'bpmn'    : get_k_bpmn_triple (5 字段, "pool or lane")
# 'flowgen' : get_k_flowgen_triple (5 字段, "group or container")
DS_PROMPT_TYPE = {
    "cbd": "cbdfcb", "fcb": "cbdfcb", "flowvqa": "cbdfcb",
    "fca": "cbdfcb", "flowlearn": "cbdfcb",
    "bpmn": "bpmn",
    "flowgen_easy": "flowgen", "flowgen_medium": "flowgen", "flowgen_hard": "flowgen",
}
DS_BPMN_FLAG = {k: (v == "bpmn") for k, v in DS_PROMPT_TYPE.items()}  # 向后兼容

# 原图绘制参数
def get_draw_params(dataset: str) -> Tuple[int, int, float, float]:
    """(line_width, margin, font_scale_mul, box_area_ratio)
       font_scale_mul: 字号缩放倍率
       box_area_ratio: bbox 面积缩放（围绕中心收缩；1.0=不变；0.5=面积减半=边长 sqrt(0.5)）
    """
    if dataset == "cbd":
        return 3, 5, 1.0, 1.0
    if dataset == "flowvqa":
        # flowvqa 推理 margin=5; K-box 边长×0.75 (area_ratio=0.5625), 数字字号×0.5
        return 3, 5, 0.5, 0.5625
    if dataset == "flowlearn":
        # Match the K=1 annotation: expand the arrowhead bbox by 5 px on each side.
        return 3, 5, 1.0, 1.0
    if dataset == "fcb":
        return 3, 0, 1.0, 1.0
    if dataset == "bpmn":
        # 用户要求：arrowhead 矩形面积缩到原来 50%，数字大小缩到原来 50%
        return 3, 0, 0.375, 0.5  # 0.375 = 之前 0.75 × 0.5
    if dataset.startswith("flowgen_"):
        # 用户要求：数字字号 40%，位置在左上侧（不在正上方）
        return 3, 0, 0.4, 1.0
    return 3, 0, 1.0, 1.0


def get_split_config(dataset: str, split: str) -> Dict:
    """
    返回当前 (dataset, split) 的路径配置。
    keys:
      training_all_json: K=1 训练数据
      folder_dir: per-folder merged JSON 所在父目录
      merged_filename: lambda folder -> json filename
      orig_img_dir: 原图所在父目录（folder.png/jpg）
      orig_img_ext: list of extensions to try
      out_data_subdir: 在 folder_dir 下放新图片的子目录名
    """
    if dataset == "cbd":
        if split == "train":
            return {
                "training_all_json": DATA_ROOT / "cbd_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "cbd",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/train_img",
                "orig_img_ext": [".png", ".jpg"],
            }
        elif split == "val":
            return {
                "training_all_json": DATA_ROOT / "cbd_val_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "cbd_val",
                "merged_filename": lambda f: f"{f}.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/val_img",
                "orig_img_ext": [".png", ".jpg"],
            }
        elif split == "test":
            # CBD test 没有 per-image merged JSON，需要从 K=1 baseline 的 arrow_triplets.json 拿 bbox
            # 但 K=1 baseline 是 detection-based。这里我们暂不为 CBD 生成 K-test 数据，
            # 而是通过 K-aware 推理直接在 inference 时分组。
            raise ValueError("CBD test 不支持离线生成 K 数据。请在 K 推理脚本中实时分组。")
    elif dataset == "fcb":
        if split == "train":
            return {
                "training_all_json": DATA_ROOT / "fcb_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "fcb",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/train",
                "orig_img_ext": [".png", ".jpg"],
            }
        elif split == "val":
            return {
                "training_all_json": DATA_ROOT / "fcb_val_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "fcb_val",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/val",
                "orig_img_ext": [".png", ".jpg"],
            }
        elif split == "test":
            return {
                "training_all_json": None,  # 见下文 build_test_for_fcb
                "folder_dir": DATA_ROOT / "fcb_test",
                "merged_filename": lambda f: f"{f}.json",  # fcb_test 用未 _merged 的版本
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/test",
                "orig_img_ext": [".png", ".jpg"],
            }
    elif dataset.startswith("flowgen_"):
        # dataset = flowgen_easy / flowgen_medium / flowgen_hard
        difficulty = dataset.split("_", 1)[1]  # easy / medium / hard
        if split == "train":
            return {
                "training_all_json": DATA_ROOT / f"flowgen_train_{difficulty}.json",
                "folder_dir": DATA_ROOT / f"flowgen_train_{difficulty}",
                "merged_filename": lambda f: f"{f}_merged.json",
                # flowgen 原图依 source 类型分散在 flowgen/train_{source}_{difficulty}_png/
                "orig_img_dir": None,  # 用 find_orig_image_flowgen
                "orig_img_ext": [".png"],
                "flowgen_split": "train",
                "flowgen_difficulty": difficulty,
            }
        elif split == "val":
            return {
                "training_all_json": DATA_ROOT / f"flowgen_val_{difficulty}.json",
                "folder_dir": DATA_ROOT / f"flowgen_val_{difficulty}",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": None,
                "orig_img_ext": [".png"],
                "flowgen_split": "val",
                "flowgen_difficulty": difficulty,
            }
        elif split == "test":
            return {
                "training_all_json": DATA_ROOT / f"flowgen_test_{difficulty}.json",
                "folder_dir": DATA_ROOT / f"flowgen_test_{difficulty}",
                "merged_filename": lambda f: f"{f}.json",  # test 没有 _merged
                "orig_img_dir": None,
                "orig_img_ext": [".png"],
                "flowgen_split": "test",
                "flowgen_difficulty": difficulty,
            }
    elif dataset == "bpmn":
        # 注意：BPMN 的 bbox 坐标对应 new_*_images 这套小尺寸渲染（≈1697×1234），
        # 不是 *_images（≈7364×5358 的大尺寸渲染）。
        if split == "train":
            return {
                "training_all_json": DATA_ROOT / "bpmn_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "bpmn",
                "merged_filename": lambda f: f"{f}.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/bpmn/new_train_images",
                "orig_img_ext": [".png"],
            }
        elif split == "val":
            return {
                "training_all_json": DATA_ROOT / "bpmn_val_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "bpmn",
                "merged_filename": lambda f: f"{f}.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/bpmn/new_dev_images",
                "orig_img_ext": [".png"],
            }
        elif split == "test":
            return {
                "training_all_json": DATA_ROOT / "bpmn_test_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "bpmn_test",
                "merged_filename": lambda f: f"{f}.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/bpmn/new_test_images",
                "orig_img_ext": [".png"],
            }
    elif dataset == "flowvqa":
        # flowvqa 无独立 val arrow 文件, 用 split_flowvqa_arrow.py 切的 9:1 (train/val)
        # folder_dir = flowvqa/, merged = {f}_merged.json, 原图在 Data/A. Main Set Flowchart Images/
        if split == "train":
            return {
                "training_all_json": DATA_ROOT / "flowvqa_arrow_train.json",
                "folder_dir": DATA_ROOT / "flowvqa",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": DATA_ROOT / "flowvqa",  # find_orig_image 嵌套查找 {folder}/{folder}.png (与 K1 标注图同源)
                "orig_img_ext": [".png"],
            }
        elif split == "val":
            return {
                "training_all_json": DATA_ROOT / "flowvqa_arrow_val.json",
                "folder_dir": DATA_ROOT / "flowvqa",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": DATA_ROOT / "flowvqa",
                "orig_img_ext": [".png"],
            }
    elif dataset == "fca":
        # FC_A 手写框图；箭头离散无重叠 → 不需要 merged/K=1 答案文件，
        # 直接用 per-folder 的非 merged json ({folder}.json, 含 bbox+start/end/condition) 派生答案。
        # 整页原图 = inkml 渲染的 PNG (Block-Diagram/.../FC_A/train_img/{folder}.png, 坐标已对齐)。
        # train/val 在 folder 级切分（--folder-list 限定），避免 entry 级泄漏。
        if split in ("train", "val"):
            return {
                "training_all_json": None,  # 派生模式：从 per-folder json 取答案
                "folder_dir": DATA_ROOT / "fca",
                "merged_filename": lambda f: f"{f}.json",  # 非 merged
                "orig_img_dir": PROJECT_ROOT / "Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_A/train_img",
                "orig_img_ext": [".png"],
            }
    elif dataset == "flowlearn":
        # FlowLearn (mermaid 生成)；K=1 _all 无独立 val，用 split_train_val.py 切 90/10 (seed42, entry级)
        # 原图在 flowlearn/mermaid_word/jpeg/{folder}.jpeg；folder_dir/folder/{folder}_merged.json
        if split in ("train", "val"):
            return {
                "training_all_json": DATA_ROOT / f"flowlearn_{split}_arrow_training_all.json",
                "folder_dir": DATA_ROOT / "flowlearn",
                "merged_filename": lambda f: f"{f}_merged.json",
                "orig_img_dir": PROJECT_ROOT / "Dataset/flowlearn/mermaid_word/jpeg",
                "orig_img_ext": [".jpeg"],
            }
    raise ValueError(f"Unknown dataset/split: {dataset}/{split}")


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────
ID_RE = re.compile(r"^(.+)_(\d+)$")


def parse_entry_id(entry_id: str) -> Optional[Tuple[str, int]]:
    """'Connect (36)_1' → ('Connect (36)', 1)"""
    m = ID_RE.match(entry_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def find_orig_image(orig_img_dir: Path, folder: str, exts: List[str]) -> Optional[Path]:
    for ext in exts:
        p = orig_img_dir / f"{folder}{ext}"
        if p.exists():
            return p
    # fallback: 嵌套 {folder}/{folder}{ext} (flowvqa 等: 干净原图在 folder_dir/{folder}/{folder}.png,
    # 与 K1 标注图同源同分辨率; Data/ 下的高清原图是不同渲染, bbox 坐标无法映射)
    for ext in exts:
        p = orig_img_dir / folder / f"{folder}{ext}"
        if p.exists():
            return p
    return None


_FLOWGEN_BASE = PROJECT_ROOT / "Dataset/flowgen"

def find_orig_image_flowgen(folder: str, split: str, difficulty: str) -> Optional[Path]:
    """flowgen folder name pattern: '{source}_{difficulty}_{source}_{NNN}'
       原图位置: flowgen/{split}_{source}_{difficulty}_png/{source}_{NNN}.png
       split={train|test} ; val 也用 train 目录（val 是从 train 切的）

       注意: val 数据集没有独立的 png 目录，需要遍历 train_{source}_{difficulty}_png 查找。
       由于 val/{folder} 与 train 共享相同的 _png，使用 train_xxx_png 即可。
    """
    # 拆 folder: '{source}_{difficulty}_{rest}'
    # rest 通常是 '{source}_{NNN}'
    marker = f"_{difficulty}_"
    if marker not in folder:
        return None
    source, rest = folder.split(marker, 1)
    # 期待 rest 形如 'graphviz_282' → 取整个 rest 作为 png 文件名 stem
    # 不同 split 有不同 png 目录前缀
    if split == "test":
        cand_dirs = [
            _FLOWGEN_BASE / f"test_img_{difficulty}",
            _FLOWGEN_BASE / f"test_img_{difficulty}_orig",
        ]
    else:
        cand_dirs = [_FLOWGEN_BASE / f"train_{source}_{difficulty}_png"]
    for d in cand_dirs:
        p = d / f"{rest}.png"
        if p.exists():
            return p
    return None


def _shrink_bbox(bbox: List[float], area_ratio: float) -> List[float]:
    """围绕中心按 area_ratio 收缩。area_ratio=0.5 → 边长 ×sqrt(0.5)"""
    if area_ratio >= 1.0:
        return bbox
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = area_ratio ** 0.5
    hw = (x2 - x1) / 2.0 * scale
    hh = (y2 - y1) / 2.0 * scale
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def draw_k_box_image(
    src_img_path: Path,
    bboxes_with_label: List[Tuple[List[float], int]],
    out_path: Path,
    line_width: int = 3,
    margin: int = 0,
    font_scale_mul: float = 1.0,
    box_area_ratio: float = 1.0,
) -> bool:
    """
    bboxes_with_label: [(bbox[x1,y1,x2,y2], label_idx_starting_from_1), ...]
    画蓝框 + 紧邻框左上角的数字标签。
    """
    img = cv2.imread(str(src_img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    blue = (255, 0, 0)  # BGR

    # 1) 先画所有蓝框（先按 area 收缩，再 margin 扩展）
    for bbox, _ in bboxes_with_label:
        shr = _shrink_bbox(bbox, box_area_ratio)
        x1, y1, x2, y2 = map(int, shr)
        if margin > 0:
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
        cv2.rectangle(img, (x1, y1), (x2, y2), blue, line_width)

    # 2) 数字标签：紧贴框左上、透明底
    img_diag = (h ** 2 + w ** 2) ** 0.5
    font_scale = max(0.6, min(1.6, img_diag / 1500.0)) * font_scale_mul
    font_thickness = max(2, int(round(font_scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for bbox, label in bboxes_with_label:
        shr = _shrink_bbox(bbox, box_area_ratio)
        x1, y1, x2, y2 = map(int, shr)
        if margin > 0:
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
        text = str(label)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        # 左上侧：文字右边缘紧贴 box 左侧（外）；基线对齐 box 上边
        gap = 2  # box 与文字之间的水平间距
        text_x = max(0, x1 - tw - gap)
        text_y = y1  # 基线 = box 上边
        # 如顶部空间不够（基线减去 ascent 仍 < 0），则置于框内左侧靠上
        if text_y - th < 0:
            text_y = y1 + th + 1
            text_x = x1 + 1  # 框内左上角
        cv2.putText(img, text, (text_x, text_y), font, font_scale, blue, font_thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def slide_window_groups(items: List, k: int) -> List[List]:
    """[a,b,c,d,e], k=2 -> [[a,b],[c,d],[e]]"""
    return [items[i:i + k] for i in range(0, len(items), k)]


def cnk_capped_groups(items: List, k: int, cap_multiplier: Optional[int] = None,
                      seed: int = 42) -> List[List]:
    """C(n,k) 组合 capped 到 cap_multiplier × ceil(n/k)。

    - cap_multiplier=None 时默认为 k（即 k×cnk：cap = k * ceil(n/k)），
      使各 K 的每图组数都 ≈ n（按箭头数等量配额，跨 K 公平比较）。
      显式传入 --cnk-cap-mul 仍可覆盖（如固定倍率 3）。
    - n < k：返回 [items]（一组短尾巴，与 sliding 行为一致）
    - n == k：唯一一组
    - 否则：列出所有 C(n,k) 组合；如超过 cap 则随机采样 cap 个。
    """
    n = len(items)
    if n < k:
        return [list(items)]
    if cap_multiplier is None:
        cap_multiplier = k  # 默认 k×cnk（K=2→倍率2, K=3→倍率3）
    sliding_count = (n + k - 1) // k  # ceil(n/k)
    cap = cap_multiplier * sliding_count
    all_combos = list(combinations(range(n), k))
    if len(all_combos) <= cap:
        chosen = all_combos
    else:
        rng = random.Random(seed)
        chosen = rng.sample(all_combos, cap)
    return [[items[i] for i in combo] for combo in chosen]


def build_k_answer(per_arrow_answers: List[str]) -> str:
    """将多个 K=1 答案块拼成 K 答案。
       每块前加 'arrow_<i>:'，块间空行分隔。"""
    blocks = []
    for i, ans in enumerate(per_arrow_answers, 1):
        blocks.append(f"arrow_{i}:\n{ans.strip()}")
    return "\n\n".join(blocks)


def format_list_text(data_list):
    """复刻 gen_data/format_data.py:format_list_text —— 保证派生答案与 K=1 一致。
       空/None → ['None']；列表 → 去空白后的非空串，全空则 ['None']。"""
    if not data_list:
        return ["None"]
    if isinstance(data_list, list):
        processed = [str(t).strip() for t in data_list if t]
        return processed if processed else ["None"]
    return [str(data_list).strip()]


def format_arrow_answer_from_json(arrow: Dict) -> str:
    """从 per-folder json 的 arrow 条目派生 K=1 答案文本（与 format_data.py 一致）。
       用于 FCA：箭头离散无需 merged，直接用 {folder}.json 的 start/end/condition。"""
    start_text = " || ".join(format_list_text(arrow.get("start_point", [])))
    end_text = " || ".join(format_list_text(arrow.get("end_point", [])))
    cond_text = " || ".join(format_list_text(arrow.get("condition", [])))
    return f"source_nodes: {start_text}\nend_nodes: {end_text}\ncondition: {cond_text}"


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def build_k_data_one_split(dataset: str, split: str, k: int,
                           strategy: str = "sliding",
                           cnk_cap_mul: Optional[int] = None,
                           cnk_seed: int = 42,
                           verbose: bool = True,
                           folder_list_path: Optional[str] = None,
                           overwrite_images: bool = False) -> List[Dict]:
    cfg = get_split_config(dataset, split)
    training_all_json = cfg["training_all_json"]
    folder_dir = cfg["folder_dir"]
    merged_filename_fn = cfg["merged_filename"]
    orig_img_dir = cfg["orig_img_dir"]
    orig_img_ext = cfg["orig_img_ext"]

    line_width, margin, font_scale_mul, box_area_ratio = get_draw_params(dataset)
    prompt_type = DS_PROMPT_TYPE.get(dataset, "cbdfcb")
    if prompt_type == "bpmn":
        prompt_fn = get_k_bpmn_triple
    elif prompt_type == "flowgen":
        prompt_fn = get_k_flowgen_triple
    else:
        prompt_fn = get_k_triple
    is_flowgen = prompt_type == "flowgen"
    flowgen_split = cfg.get("flowgen_split")
    flowgen_difficulty = cfg.get("flowgen_difficulty")

    # 加载 K=1 答案
    answer_map: Dict[Tuple[str, int], str] = {}
    if training_all_json is not None and training_all_json.exists():
        k1_data = json.load(open(training_all_json, "r", encoding="utf-8"))
        for entry in k1_data:
            parsed = parse_entry_id(entry["id"])
            if not parsed:
                continue
            folder, arrow_id = parsed
            answer_map[(folder, arrow_id)] = entry["conversations"][1]["value"]
        if verbose:
            print(f"  Loaded {len(k1_data)} K=1 entries from {training_all_json.name}")
    else:
        if verbose:
            print(f"  ⚠ training_all_json not found ({training_all_json}); will derive answers from merged JSON")

    # 列出本 split 涉及的 folders
    if answer_map:
        folders = sorted({f for (f, _) in answer_map.keys()})
    else:
        folders = sorted([p.name for p in folder_dir.iterdir() if p.is_dir()])

    # 可选：用 folder 名单文件限定本 split 处理的 folders（FCA 等 folder 级 train/val 切分用）
    if folder_list_path:
        with open(folder_list_path, "r", encoding="utf-8") as f:
            allow = {ln.strip() for ln in f if ln.strip()}
        folders = [fdr for fdr in folders if fdr in allow]
        if verbose:
            print(f"  Folder-list filter: {folder_list_path} → {len(folders)} folders")

    if verbose:
        print(f"  Folders to process: {len(folders)}")

    out_entries: List[Dict] = []
    n_imgs_drawn = 0
    n_skipped_no_orig = 0
    n_skipped_no_merged = 0
    n_skipped_no_answer = 0
    n_groups_total = 0

    annotated_subdir = folder_dir / f"_annotated_images_k{k}"  # 全 split 共用一个根
    annotated_subdir.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        # 加载 merged JSON
        mj_path = folder_dir / folder / merged_filename_fn(folder)
        if not mj_path.exists():
            n_skipped_no_merged += 1
            continue
        try:
            merged = json.load(open(mj_path, "r", encoding="utf-8"))
        except Exception as e:
            n_skipped_no_merged += 1
            if verbose:
                print(f"  ⚠ merged JSON parse error: {mj_path}: {e}")
            continue
        arrows = sorted(merged.get("arrows", []), key=lambda a: a.get("arrow_id", 0))
        if not arrows:
            continue

        # FCA 等无需 K=1 答案文件的数据集：直接从 per-folder json 派生每箭答案
        # (training_all_json=None 时 answer_map 为空，靠 folder_answer_map 兜底)
        folder_answer_map: Dict = {}
        if training_all_json is None:
            for a in arrows:
                aid = a.get("arrow_id")
                if aid is None:
                    continue
                folder_answer_map[aid] = format_arrow_answer_from_json(a)

        # 仅保留有 bbox 且 (folder,arrow_id) 在 answer_map 中 的箭头
        # flowgen 用 'arrowhead_bbox'，其它用 'matched_arrowhead_bbox'
        usable = []
        for a in arrows:
            aid = a.get("arrow_id")
            bbox = a.get("matched_arrowhead_bbox") or a.get("arrowhead_bbox")
            if aid is None or bbox is None:
                continue
            # 训练/val 必须有答案；test 时如果 training_all_json 不存在则跳过
            if (folder, aid) not in answer_map and training_all_json is not None:
                continue
            usable.append((aid, bbox))
        if not usable:
            n_skipped_no_answer += 1
            continue

        # 原图（flowgen 走专门 lookup；其它用通用 find）
        if is_flowgen:
            orig_img = find_orig_image_flowgen(folder, flowgen_split, flowgen_difficulty)
        else:
            orig_img = find_orig_image(orig_img_dir, folder, orig_img_ext)
        if orig_img is None:
            n_skipped_no_orig += 1
            if verbose and n_skipped_no_orig <= 3:
                print(f"  ⚠ no orig img for {folder} in {orig_img_dir}")
            continue

        # 分组策略
        if strategy == "cnk":
            groups = cnk_capped_groups(usable, k, cnk_cap_mul, cnk_seed)
        else:
            groups = slide_window_groups(usable, k)
        for group in groups:
            n = len(group)
            # 1) 绘图
            ids_str = "-".join(str(aid) for aid, _ in group)
            out_img_path = annotated_subdir / folder / f"{folder}_arrows_{ids_str}.png"
            if overwrite_images or not out_img_path.exists():
                bboxes_with_label = [(bbox, i + 1) for i, (_, bbox) in enumerate(group)]
                ok = draw_k_box_image(
                    orig_img, bboxes_with_label, out_img_path,
                    line_width=line_width, margin=margin,
                    font_scale_mul=font_scale_mul,
                    box_area_ratio=box_area_ratio,
                )
                if not ok:
                    if verbose:
                        print(f"  ⚠ failed to draw {out_img_path}")
                    continue
                n_imgs_drawn += 1
            # 2) 构 prompt + answer
            user_value = prompt_fn(n)
            per_arrow_answers = []
            missing = False
            for aid, _ in group:
                if (folder, aid) in answer_map:
                    per_arrow_answers.append(answer_map[(folder, aid)])
                elif aid in folder_answer_map:
                    per_arrow_answers.append(folder_answer_map[aid])
                else:
                    missing = True
                    break
            if missing:
                n_skipped_no_answer += 1
                continue
            assistant_value = build_k_answer(per_arrow_answers)
            entry_id = f"{folder}__k{k}__{ids_str}"
            out_entries.append({
                "id": entry_id,
                "image": str(out_img_path),
                "conversations": [
                    {"from": "user", "value": user_value},
                    {"from": "assistant", "value": assistant_value},
                ],
            })
            n_groups_total += 1

    if verbose:
        print(f"  Generated {len(out_entries)} K={k} entries "
              f"(images drawn={n_imgs_drawn}, skip_no_orig={n_skipped_no_orig}, "
              f"skip_no_merged={n_skipped_no_merged}, skip_no_answer={n_skipped_no_answer})")
    return out_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="cbd | fcb | bpmn | all (逗号分隔)")
    parser.add_argument("--split", type=str, required=True,
                        help="train | val | test | all (逗号分隔)")
    parser.add_argument("--k", type=str, required=True,
                        help="单个 K 或逗号分隔多个，如 2 或 2,3")
    parser.add_argument("--out-dir", type=Path,
                        default=DATA_ROOT,
                        help="输出 JSON 文件目录")
    parser.add_argument("--strategy", choices=["sliding", "cnk"], default="sliding",
                        help="分组策略：sliding=滑窗不重叠；cnk=C(n,K) 上限采样")
    parser.add_argument("--cnk-cap-mul", type=int, default=None,
                        help="cnk 上限倍率 (cap = mul * ceil(n/k))；默认 None=k→ k×cnk (K=2→2, K=3→3)")
    parser.add_argument("--cnk-seed", type=int, default=42)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件名后缀，用于和 sliding 版本区分。如 '_cnk'")
    parser.add_argument("--folder-list", type=str, default=None,
                        help="folder 名单文件 (一行一个)，限定本 split 只处理名单内 folders。"
                             "FCA 等 folder 级 train/val 切分用。")
    parser.add_argument("--overwrite-images", action="store_true",
                        help="Redraw existing K-box images with the current drawing parameters.")
    args = parser.parse_args()

    datasets = ["cbd", "fcb", "bpmn"] if args.dataset == "all" else args.dataset.split(",")
    splits = ["train", "val", "test"] if args.split == "all" else args.split.split(",")
    ks = [int(x) for x in args.k.split(",")]

    for ds in datasets:
        for sp in splits:
            for k in ks:
                if ds == "cbd" and sp == "test":
                    print(f"\n[skip] {ds}/{sp}/K={k}: CBD test 没有 K=1 训练数据，K-test 在推理时实时分组")
                    continue
                if ds == "fcb" and sp == "test":
                    print(f"\n[skip] {ds}/{sp}/K={k}: FCB test 没有 K=1 训练数据，K-test 在推理时实时分组")
                    continue
                if ds in ("fca", "flowlearn") and sp == "test":
                    print(f"\n[skip] {ds}/{sp}/K={k}: {ds} test 没有 K=1 训练数据，K-test 在推理时实时分组")
                    continue
                if ds.startswith("flowgen_") and sp == "test":
                    # flowgen test JSON 没有 _merged 结构，且推理时实时分组即可
                    print(f"\n[skip] {ds}/{sp}/K={k}: FlowGen test 在推理时实时分组")
                    continue
                print(f"\n=== {ds} / split={sp} / K={k} / strategy={args.strategy} ===")
                try:
                    entries = build_k_data_one_split(
                        ds, sp, k,
                        strategy=args.strategy,
                        cnk_cap_mul=args.cnk_cap_mul,
                        cnk_seed=args.cnk_seed,
                        folder_list_path=args.folder_list,
                        overwrite_images=args.overwrite_images,
                    )
                except Exception as e:
                    print(f"  ✗ failed: {e}")
                    import traceback; traceback.print_exc()
                    continue
                suffix = args.out_suffix
                out_path = args.out_dir / f"{ds}_k{k}_{sp}{suffix}_arrow_training.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                print(f"  → {out_path}  ({len(entries)} entries)")


if __name__ == "__main__":
    main()
