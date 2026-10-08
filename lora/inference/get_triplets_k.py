"""
get_triples_sft_v1_k.py — K-arrow inference.

每张图按 arrow_id 升序滑窗 K 个一组（最后 <K 单独一组），渲染 K-box 图像，
查询 K-trained Qwen-VL，解析 N 个编号 block 拆回单箭头 triples，
按图聚合写入 output_dir/<image_stem>/arrow_triplets.json。

为节省计算，bbox 直接从已有 K=1 baseline 推理输出目录中读取
（每个 <image_stem>/arrow_triplets.json 含 arrow_bbox 字段）。

Usage:
  python -m lora.inference.get_triplets_k \
    --bbox-source-dir <K=1 baseline output dir> \
    --image-dir <original test images> \
    --output-dir <K output dir> \
    --adapter-path <K LoRA ckpt> \
    --base-model-path /data/.../Qwen3-VL-4B-Instruct \
    --gpus 0,1,2,3 \
    --k 2
"""
import os
import json
import re
import math
import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

from lora.k_prompt import (
    get_k_triple, get_k_bpmn_triple, get_k_flowgen_triple,
    get_triple, get_bpmn_triple, get_flowgen_triple,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-dir", required=True, help="原始测试图片目录")
    p.add_argument("--bbox-source-dir", required=True,
                   help="K=1 baseline 推理输出目录，含 <image_stem>/arrow_triplets.json 的 arrow_bbox 字段")
    p.add_argument("--k", type=int, required=True, help="K 值（每次查询的箭头数量）")
    p.add_argument("--adapter-path", default=None, help="K-trained LoRA adapter 路径")
    p.add_argument("--base-model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--is-bpmn", action="store_true", help="使用 BPMN 5字段 prompt + parser")
    p.add_argument("--is-flowgen", action="store_true", help="使用 FlowGen 5字段 prompt + parser (与 BPMN 相同字段，但 prompt 不同)")
    p.add_argument("--result-filename", default="arrow_triplets.json")
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--draw-margin", type=int, default=-1, help="-1 = 数据集默认；cbd/bpmn=5, fcb=0")
    p.add_argument("--draw-line-width", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="多箭头输出更长，必须足够大")
    p.add_argument("--max-images", type=int, default=0,
                   help="Limit evaluation to this many images; 0 uses all images.")
    p.add_argument("--uniform-sample", action="store_true",
                   help="Select max-images evenly across the sorted image list.")
    p.add_argument("--bbox-source-only", action="store_true",
                   help="Only evaluate images that have a saved K=1 bbox-source result.")
    p.add_argument("--timing-output", default=None,
                   help="Write inference-only timing JSON (model loading excluded).")
    p.add_argument("--few-shot", type=int, choices=(0, 3), default=0,
                   help="FlowGen multi-arrow prompt demonstrations (0 or 3).")
    return p.parse_args()


def infer_margin_from_image_dir(image_dir: str) -> int:
    s = image_dir.lower()
    if "fc_b" in s or "fcb" in s:
        return 0
    if "bpmn" in s:
        return 0  # BPMN: 不外扩 margin
    if "flowgen" in s:
        return 0  # FlowGen: 不外扩 margin
    return 5  # cbd / others


def infer_box_area_ratio(is_bpmn: bool, is_flowgen: bool = False, is_flowvqa: bool = False) -> float:
    """与 gen_k_arrow_data.get_draw_params 对齐"""
    if is_bpmn:
        return 0.5
    if is_flowgen:
        return 1.0  # FlowGen 不缩 box
    if is_flowvqa:
        return 0.5625  # 边长×0.75
    return 1.0


def infer_font_scale_mul(is_bpmn: bool, is_flowgen: bool = False, is_flowvqa: bool = False) -> float:
    """与 gen_k_arrow_data.get_draw_params 对齐"""
    if is_bpmn:
        return 0.375
    if is_flowgen:
        return 0.4  # FlowGen 字号 0.4
    if is_flowvqa:
        return 0.5  # 数字字号×0.5
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 渲染
# ─────────────────────────────────────────────────────────────────────────────
def _shrink_bbox_pil(bbox, area_ratio: float):
    """围绕中心按 area_ratio 收缩 (area_ratio=0.5 → 边长 ×sqrt(0.5))。"""
    if area_ratio >= 1.0:
        return bbox
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = area_ratio ** 0.5
    hw = (x2 - x1) / 2.0 * scale
    hh = (y2 - y1) / 2.0 * scale
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def draw_k_box_on_image(
    image_pil: Image.Image,
    bboxes_with_label: List[Tuple[List[float], int]],
    line_width: int = 3,
    margin: int = 0,
    box_area_ratio: float = 1.0,
    font_scale_mul: float = 1.0,
) -> Image.Image:
    """返回标好 K 个蓝框 + 数字标签的新图（PIL）。
    与 gen_k_arrow_data.py 中的 draw 完全对齐（face/位置/缩放）。"""
    img = image_pil.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 自适应字号（与 gen_k_arrow_data.draw_k_box_image 对齐）
    diag = (w * w + h * h) ** 0.5
    base = max(0.6, min(1.6, diag / 1500.0)) * font_scale_mul
    # cv2 用 font_scale；PIL 需对应像素 font_size：~ scale × 24
    font_size = max(8, int(round(base * 22)))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    blue = (0, 0, 255)

    # 画框
    for bbox, _ in bboxes_with_label:
        shr = _shrink_bbox_pil(bbox, box_area_ratio)
        x1, y1, x2, y2 = map(int, shr)
        if margin > 0:
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
        draw.rectangle([x1, y1, x2, y2], outline=blue, width=line_width)

    # 数字标签：左上侧——文字右边缘紧贴 box 左侧外，顶端对齐 box 上边
    # (对齐 gen_k_arrow_data.py 的 cv2 实现)
    gap = 2
    for bbox, label in bboxes_with_label:
        shr = _shrink_bbox_pil(bbox, box_area_ratio)
        x1, y1, x2, y2 = map(int, shr)
        if margin > 0:
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        text = str(label)
        try:
            tbbox = draw.textbbox((0, 0), text, font=font)
            tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
        except Exception:
            tw, th = int(font_size * 0.6 * len(text)), int(font_size)
        # PIL draw.text 的 (x,y) 是文本左上角（不是基线）
        tx = max(0, x1 - tw - gap)
        ty = max(0, y1 - th)  # 文本顶端对齐 box 上边（即文本底端在 y1）
        if ty < 0:
            # 顶部空间不够：放框内左上
            ty = y1 + 1
            tx = x1 + 1
        draw.text((tx, ty), text, fill=blue, font=font)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 解析 K-block 输出
# ─────────────────────────────────────────────────────────────────────────────
ARROW_BLOCK_RE = re.compile(r"arrow_(\d+)\s*:\s*", re.IGNORECASE)


def split_k_blocks(raw: str, expected_n: int) -> List[str]:
    """把 'arrow_1:\n...\narrow_2:\n...' 拆成 list of N block content strings."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    matches = list(ARROW_BLOCK_RE.finditer(raw))
    blocks = []
    if not matches:
        # 兼容退化：把整段当作 arrow_1
        blocks.append(raw)
    else:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            blocks.append(raw[start:end].strip())
    # pad to expected_n
    while len(blocks) < expected_n:
        blocks.append("")
    return blocks[:expected_n]


def parse_block_cbdfcb(block: str) -> Dict:
    cleaned = block.strip()
    source = re.search(r"source_nodes\s*:\s*(.+?)(?=\n\s*(?:end_nodes|condition)\s*:|$)",
                       cleaned, re.IGNORECASE | re.DOTALL)
    end = re.search(r"end_nodes\s*:\s*(.+?)(?=\n\s*condition\s*:|$)",
                    cleaned, re.IGNORECASE | re.DOTALL)
    cond = re.search(r"condition\s*:\s*(.+?)(?=$)", cleaned, re.IGNORECASE | re.DOTALL)
    return {
        "source_raw": source.group(1).strip() if source else "[UNKNOWN]",
        "end_raw": end.group(1).strip() if end else "[UNKNOWN]",
        "cond_raw": cond.group(1).strip() if cond else "None",
    }


def parse_block_bpmn(block: str) -> Dict:
    def f(name):
        # capture until next field or end
        m = re.search(rf"{name}\s*:\s*(.+?)(?=\n\s*(?:source_node|source_location|condition|target_node|target_location)\s*:|$)",
                      block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else "None"
    return {
        "source_node": f("source_node"),
        "source_location": f("source_location"),
        "condition": f("condition"),
        "target_node": f("target_node"),
        "target_location": f("target_location"),
    }


def cbdfcb_block_to_triplets(parsed: Dict, arrow_info: Dict) -> List[Dict]:
    """ 把 '||'-分隔 的源/终/条件拆成多个 triplet。"""
    sources = [s.strip() for s in parsed["source_raw"].split("||")]
    ends = [s.strip() for s in parsed["end_raw"].split("||")]
    conds = [s.strip() for s in parsed["cond_raw"].split("||")]
    n = max(len(sources), len(ends), len(conds))
    # pad
    while len(sources) < n: sources.append(sources[-1] if sources else "[UNKNOWN]")
    while len(ends) < n: ends.append(ends[-1] if ends else "[UNKNOWN]")
    while len(conds) < n: conds.append("None")
    triplets = []
    for s, e, c in zip(sources, ends, conds):
        if c.lower() in ("none", "", "null"):
            c = None
        triplets.append({
            "arrow_id": arrow_info["arrow_id"],
            "arrow_label": arrow_info["arrow_label"],
            "source": s,
            "end": e,
            "condition": c,
            "arrow_bbox": arrow_info["arrow_bbox"],
            "raw_vqa_output": arrow_info["raw_block"],
            "full_vqa_output": arrow_info.get("full_vqa_output", ""),
            "group_size": arrow_info.get("group_size", 1),
        })
    return triplets


def bpmn_block_to_triplet(parsed: Dict, arrow_info: Dict) -> List[Dict]:
    return [{
        "arrow_id": arrow_info["arrow_id"],
        "arrow_label": arrow_info["arrow_label"],
        "source_node": parsed["source_node"],
        "source_location": parsed["source_location"],
        "condition": parsed["condition"],
        "target_node": parsed["target_node"],
        "target_location": parsed["target_location"],
        "arrow_bbox": arrow_info["arrow_bbox"],
        "raw_vqa_output": arrow_info["raw_block"],
        "full_vqa_output": arrow_info.get("full_vqa_output", ""),
        "group_size": arrow_info.get("group_size", 1),
    }]


# ─────────────────────────────────────────────────────────────────────────────
# 取 bbox
# ─────────────────────────────────────────────────────────────────────────────
def load_bboxes_for_image(bbox_source_dir: str, image_stem: str) -> List[List[float]]:
    """从 K=1 baseline 推理输出读 bbox list（按 arrow_id 升序）。"""
    p = Path(bbox_source_dir) / image_stem / "arrow_triplets.json"
    if not p.exists():
        return []
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return []
    if not data:
        return []
    # 去重：同一 arrow_id 的多条 triplet 共享 bbox
    seen = {}
    for it in data:
        aid = it.get("arrow_id")
        bbox = it.get("arrow_bbox")
        if aid is not None and bbox is not None and aid not in seen:
            seen[aid] = bbox
    # 按 arrow_id 排序
    return [seen[k] for k in sorted(seen.keys())]


def slide_groups(items, k):
    return [items[i:i + k] for i in range(0, len(items), k)]


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def worker(gpu_id: int, image_list: List[str], args_dict: Dict, start_barrier, timing_queue):
    if not image_list:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    device = "cuda:0"
    print(f"[GPU {gpu_id}] Worker started. Tasks: {len(image_list)}", flush=True)

    k = args_dict["k"]
    is_bpmn = args_dict["is_bpmn"]
    is_flowgen = args_dict.get("is_flowgen", False)
    margin = args_dict["margin"]
    line_width = args_dict["line_width"]
    box_area_ratio = args_dict.get("box_area_ratio", 1.0)
    font_scale_mul = args_dict.get("font_scale_mul", 1.0)
    bbox_dir = args_dict["bbox_source_dir"]
    output_dir = args_dict["output_dir"]
    save_debug = args_dict["save_debug"]
    result_filename = args_dict["result_filename"]
    max_new_tokens = args_dict["max_new_tokens"]
    few_shot = args_dict.get("few_shot", 0)

    # 加载模型
    print(f"[GPU {gpu_id}] Loading base model {args_dict['base_model_path']}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args_dict["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    if args_dict["adapter_path"]:
        print(f"[GPU {gpu_id}] Loading LoRA {args_dict['adapter_path']}", flush=True)
        model = PeftModel.from_pretrained(model, args_dict["adapter_path"])
    model.eval()
    processor = AutoProcessor.from_pretrained(args_dict["base_model_path"])

    # Synchronize only after every worker has loaded its model and processor.
    # Therefore the measured interval excludes model-loading time.
    start_barrier.wait(timeout=600)
    inference_start = time.perf_counter()

    successful, failed = 0, 0
    for img_idx, img_path in enumerate(image_list, 1):
        try:
            stem = Path(img_path).stem
            out_d = os.path.join(output_dir, stem)
            os.makedirs(out_d, exist_ok=True)
            out_p = os.path.join(out_d, result_filename)
            if os.path.exists(out_p):
                successful += 1
                continue

            bboxes = load_bboxes_for_image(bbox_dir, stem)
            if not bboxes:
                # 写一个空文件以避免 evaluator 报错
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump([], f)
                print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] {stem}: no bboxes, wrote empty", flush=True)
                successful += 1
                continue

            image_pil = Image.open(img_path).convert("RGB")
            # arrow_id 从 1 开始
            indexed = [(i + 1, b) for i, b in enumerate(bboxes)]
            groups = slide_groups(indexed, k)

            all_triplets = []
            for g_idx, group in enumerate(groups):
                n = len(group)
                bboxes_with_label = [(b, j + 1) for j, (_, b) in enumerate(group)]
                annotated = draw_k_box_on_image(image_pil, bboxes_with_label,
                                                line_width=line_width, margin=margin,
                                                box_area_ratio=box_area_ratio,
                                                font_scale_mul=font_scale_mul)
                if save_debug:
                    dbg_dir = os.path.join(out_d, "debug")
                    os.makedirs(dbg_dir, exist_ok=True)
                    annotated.save(os.path.join(dbg_dir, f"group_{g_idx + 1}.png"))

                if k == 1:
                    if is_bpmn:
                        prompt = get_bpmn_triple()
                    elif is_flowgen:
                        prompt = get_flowgen_triple()
                    else:
                        prompt = get_triple()
                else:
                    if is_bpmn:
                        prompt = get_k_bpmn_triple(n)
                    elif is_flowgen:
                        prompt = get_k_flowgen_triple(n, few_shot=few_shot)
                    else:
                        prompt = get_k_triple(n)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": annotated},
                        {"type": "text", "text": prompt},
                    ],
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_in], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.1,
                    )
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
                vqa_out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]

                if k == 1:
                    # K=1: 整个输出就是单个 block，不需要 arrow_<i>: 前缀
                    blocks = [vqa_out]
                else:
                    blocks = split_k_blocks(vqa_out, n)
                for j, (arrow_id, bbox) in enumerate(group):
                    arrow_info = {
                        "arrow_id": arrow_id,
                        "arrow_label": f"H{arrow_id}",
                        "arrow_bbox": bbox,
                        "raw_block": blocks[j],
                        # 保存完整 VQA output 以便诊断（同 group 内 K 个 arrow 共享）
                        "full_vqa_output": vqa_out,
                        "group_size": n,
                    }
                    if is_bpmn or is_flowgen:
                        # 5 字段 parser，flowgen 与 bpmn 共用
                        parsed = parse_block_bpmn(blocks[j])
                        all_triplets.extend(bpmn_block_to_triplet(parsed, arrow_info))
                    else:
                        parsed = parse_block_cbdfcb(blocks[j])
                        all_triplets.extend(cbdfcb_block_to_triplets(parsed, arrow_info))

            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_triplets, f, indent=2, ensure_ascii=False)
            print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] {stem}: {len(all_triplets)} triplets "
                  f"({len(groups)} groups, K={k})", flush=True)
            successful += 1
            torch.cuda.empty_cache()
        except Exception as e:
            failed += 1
            print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] FAIL {img_path}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"[GPU {gpu_id}] Done: {successful} ok, {failed} fail", flush=True)
    timing_queue.put({
        "gpu_id": gpu_id,
        "seconds": time.perf_counter() - inference_start,
        "images": len(image_list),
        "successful": successful,
        "failed": failed,
    })


def main():
    args = parse_args()
    margin = args.draw_margin
    if margin < 0:
        margin = infer_margin_from_image_dir(args.image_dir)

    image_dir = Path(args.image_dir)
    images = sorted([str(p) for p in image_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"Found {len(images)} images in {image_dir}")

    if args.bbox_source_only:
        images = [p for p in images if (
            Path(args.bbox_source_dir) / Path(p).stem / args.result_filename
        ).is_file()]
        print(f"Filtered to {len(images)} images present in bbox source")

    if args.max_images > 0 and len(images) > args.max_images:
        if args.uniform_sample:
            n = args.max_images
            indices = [round(i * (len(images) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
            images = [images[i] for i in indices]
        else:
            images = images[:args.max_images]
        print(f"Selected {len(images)} images ({'uniform' if args.uniform_sample else 'first-N'})")

    gpus = [int(g) for g in args.gpus.split(",")]
    if not images:
        raise SystemExit("No matching input images were found.")
    if args.k < 1 or args.workers_per_gpu < 1 or not gpus:
        raise SystemExit("K, workers-per-GPU, and GPU count must be positive.")
    n_workers = min(len(images), len(gpus) * args.workers_per_gpu)
    # 按 worker round-robin 切分
    chunks = [[] for _ in range(n_workers)]
    for i, img in enumerate(images):
        chunks[i % n_workers].append(img)

    is_flowvqa = "flowvqa" in args.image_dir.lower()

    args_dict = {
        "output_dir": args.output_dir,
        "bbox_source_dir": args.bbox_source_dir,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "k": args.k,
        "is_bpmn": args.is_bpmn,
        "is_flowgen": args.is_flowgen,
        "margin": margin,
        "line_width": args.draw_line_width,
        "box_area_ratio": infer_box_area_ratio(args.is_bpmn, args.is_flowgen, is_flowvqa),
        "font_scale_mul": infer_font_scale_mul(args.is_bpmn, args.is_flowgen, is_flowvqa),
        "save_debug": args.save_debug,
        "result_filename": args.result_filename,
        "max_new_tokens": args.max_new_tokens,
        "few_shot": args.few_shot,
    }
    print(f"Args: {args_dict}")

    os.makedirs(args.output_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    start_barrier = mp.Barrier(n_workers)
    timing_queue = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        gpu_id = gpus[wid % len(gpus)]
        p = mp.Process(target=worker, args=(gpu_id, chunk, args_dict, start_barrier, timing_queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise SystemExit("A worker failed; timing results are incomplete.")
    worker_timings = [timing_queue.get() for _ in procs]
    inference_wall = max((x["seconds"] for x in worker_timings), default=0.0)
    total_worker_seconds = sum(x["seconds"] for x in worker_timings)
    total_worker_images = sum(x["images"] for x in worker_timings)
    mean_worker_seconds_per_image = (
        total_worker_seconds / total_worker_images if total_worker_images else 0.0
    )
    timing = {
        "num_images": len(images),
        "num_workers": n_workers,
        "inference_wall_seconds": inference_wall,
        # Observed end-to-end makespan divided by the batch size. This includes
        # static-shard tail/straggler effects and is useful for actual throughput.
        "seconds_per_image": inference_wall / len(images) if images else 0.0,
        "observed_throughput_images_per_second": (
            len(images) / inference_wall if inference_wall else 0.0
        ),
        # Weighted average of each worker's per-image compute time. This is the
        # primary scheduling-independent latency metric for comparing K values.
        "total_worker_seconds": total_worker_seconds,
        "mean_worker_seconds_per_image": mean_worker_seconds_per_image,
        # Tail-free throughput estimate if the same workers were perfectly
        # load-balanced (e.g. a shared dynamic task queue or a much larger set).
        "ideal_balanced_seconds_per_image": (
            mean_worker_seconds_per_image / n_workers if n_workers else 0.0
        ),
        "ideal_balanced_throughput_images_per_second": (
            n_workers / mean_worker_seconds_per_image
            if mean_worker_seconds_per_image else 0.0
        ),
        "estimated_balanced_wall_seconds_900_images": (
            900 * mean_worker_seconds_per_image / n_workers if n_workers else 0.0
        ),
        "model_loading_excluded": True,
        "uniform_sample": bool(args.uniform_sample),
        "workers": worker_timings,
        "sample_stems": [Path(p).stem for p in images],
    }
    print(
        f"Inference-only timing: wall={inference_wall:.3f}s, "
        f"observed={timing['seconds_per_image']:.4f}s/image, "
        f"worker-mean={mean_worker_seconds_per_image:.4f}s/image, "
        f"ideal-balanced={timing['ideal_balanced_seconds_per_image']:.4f}s/image"
    )
    if args.timing_output:
        Path(args.timing_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timing_output, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
    print("All workers finished.")


if __name__ == "__main__":
    main()
