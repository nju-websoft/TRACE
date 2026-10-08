"""
MiniCPM-V-4_5 arrow-per-image inference script (counterpart of get_triplets_sft_trace.py).

v1 is Qwen-VL: AutoModelForImageTextToText + AutoProcessor + apply_chat_template +
              process_vision_info + model.generate + batch_decode。
MiniCPM-V4.5 needs AutoModel(trust_remote_code=True) + AutoTokenizer + model.chat().

SAM3 arrow detection, groundtruth loading, VQA prompt dispatch, and parsing all match v1,
The CLI is kept identical so shell scripts only need to swap the python module name.

Note: MiniCPM has no vLLM path, so only --backend local is supported.
"""
import os
import json
import argparse
import re
import math
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageDraw

from lora.prompt import get_triplet, get_bpmn_triplet, get_flowgen_triplet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="<OUTPUT_ROOT>/output_lora_arrow_minicpm")
    parser.add_argument("--image-dir", type=str,
                        default="<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images")
    parser.add_argument("--test-json", type=str, default="")
    parser.add_argument("--result-filename", type=str, default="arrow_triplets.json")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to the LoRA adapter")
    parser.add_argument("--base-model-path", type=str,
                        default="<MODEL_ROOT>/MiniCPM-V-4_5",
                        help="Path to the MiniCPM-V-4_5 base model")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--backend", type=str, default="local", choices=["local"],
                        help="MiniCPM only supports the local backend")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--arrow-source", type=str, default="sam",
                        choices=["sam", "sam3_ft", "yolo", "groundtruth"])
    parser.add_argument("--sam3-ft-checkpoint", type=str,
                        default="<REPO_ROOT>/detection/runs/arrowhead_detection/checkpoints/checkpoint_10.pt")
    parser.add_argument("--yolo-checkpoint", type=str, default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=2048)
    parser.add_argument("--yolo-area-filter-median-ratio", type=float, default=0.0)
    parser.add_argument("--is-bpmn", action="store_true")
    parser.add_argument("--is-flowgen", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


# ============================================================
#   Output parsing (same as v1)
# ============================================================

def extract_and_parse_vqa_output(vqa_output: str, arrow_info: Dict) -> List[Dict]:
    cleaned = re.sub(r'<think>.*?</think>', '', vqa_output, flags=re.DOTALL).strip()

    source_match = re.search(r'source_nodes:\s*(.+?)(?=\n\s*(?:end_nodes:|condition:)|$)', cleaned, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r'end_nodes:\s*(.+?)(?=\n\s*(?:condition:)|$)', cleaned, re.IGNORECASE | re.DOTALL)
    condition_match = re.search(r'condition:\s*(.+?)(?=$)', cleaned, re.IGNORECASE | re.DOTALL)

    raw_source = source_match.group(1).strip() if source_match else "[UNKNOWN]"
    raw_end = end_match.group(1).strip() if end_match else "[UNKNOWN]"
    raw_condition = condition_match.group(1).strip() if condition_match else "None"

    sources = [s.strip() for s in raw_source.split('||')]
    ends = [e.strip() for e in raw_end.split('||')]
    conditions = [c.strip() for c in raw_condition.split('||')]

    max_len = max(len(sources), len(ends), len(conditions))
    if len(sources) == 1 and max_len > 1:
        sources = sources * max_len
    if len(ends) == 1 and max_len > 1:
        ends = ends * max_len
    if len(conditions) == 1 and max_len > 1:
        conditions = conditions * max_len

    final_triplets = []
    for i in range(max_len):
        s_text = sources[i] if i < len(sources) else sources[-1]
        e_text = ends[i] if i < len(ends) else ends[-1]
        c_text = conditions[i] if i < len(conditions) else "None"
        final_condition = None if c_text.lower() == "none" else c_text
        final_triplets.append({
            "arrow_id": arrow_info["arrow_id"],
            "arrow_label": arrow_info["arrow_label"],
            "source": s_text,
            "end": e_text,
            "condition": final_condition,
            "arrow_bbox": arrow_info["arrow_bbox"],
            "raw_vqa_output": vqa_output if i == 0 else "",
        })
    return final_triplets


def extract_and_parse_bpmn_output(vqa_output: str, arrow_info: Dict) -> List[Dict]:
    cleaned = re.sub(r'<think>.*?</think>', '', vqa_output, flags=re.DOTALL).strip()

    def extract_field(name):
        m = re.search(rf'{name}:\s*(.+?)(?:\n|$)', cleaned, re.IGNORECASE)
        return m.group(1).strip() if m else "None"

    source_node = extract_field("source_node")
    source_location = extract_field("source_location")
    condition = extract_field("condition")
    target_node = extract_field("target_node")
    target_location = extract_field("target_location")

    if condition.lower() == "none":
        condition = "None"
    if source_location.lower() == "none":
        source_location = "None"
    if target_location.lower() == "none":
        target_location = "None"

    return [{
        "arrow_id": arrow_info["arrow_id"],
        "arrow_label": arrow_info["arrow_label"],
        "source_node": source_node,
        "source_location": source_location,
        "condition": condition,
        "target_node": target_node,
        "target_location": target_location,
        "arrow_bbox": arrow_info["arrow_bbox"],
        "raw_vqa_output": vqa_output,
    }]


# ============================================================
#   Drawing / debug helpers
# ============================================================

def draw_bbox_and_tag_for_vqa(image_pil: Image.Image, box: list, text_label: str) -> Image.Image:
    img_copy = image_pil.copy()
    draw = ImageDraw.Draw(img_copy)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
    return img_copy


def save_debug_image(image_pil: Image.Image, bbox: list, output_path: str):
    try:
        img_with_box = draw_bbox_and_tag_for_vqa(image_pil, bbox, "DEBUG")
        img_with_box.save(output_path)
    except Exception as e:
        print(f"Warning: Failed to save debug image {output_path}: {e}")


# ============================================================
#   Groundtruth arrow loading (same as v1)
# ============================================================

_GT_BASE_DIR = "<DATA_ROOT>/data_4_training"


def load_gt_arrows(image_stem: str, image_dir: str = "") -> list:
    base = Path(_GT_BASE_DIR)
    flowgen_diff = ""
    if image_dir:
        dir_name = Path(image_dir).name
        for d in ("easy", "medium", "hard"):
            if d in dir_name:
                flowgen_diff = d
                break

    for test_dir in sorted(base.glob("*_test*")):
        if not test_dir.is_dir():
            continue
        json_path = test_dir / image_stem / f"{image_stem}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("arrows", [])
        if test_dir.name.startswith("flowgen_test_"):
            if flowgen_diff and not test_dir.name.endswith(f"_{flowgen_diff}"):
                continue
            for sub in test_dir.iterdir():
                if sub.is_dir() and sub.name.endswith(f"_{image_stem}"):
                    json_path = sub / f"{sub.name}.json"
                    if json_path.exists():
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        return data.get("arrows", [])
    return []


# ============================================================
#   Worker: SAM3 + VQA (MiniCPM local)
# ============================================================

def creation_worker_local(gpu_id: int, image_list: List[str], args_dict: Dict):
    """Local worker: SAM3 arrow detection + MiniCPM-V-4_5 VQA."""
    if not image_list:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    device = "cuda:0"
    print(f"[GPU {gpu_id}] Local worker started. Tasks: {len(image_list)}")

    try:
        # ── Detector (skipped in groundtruth mode) ──
        sam_processor = None
        yolo_model = None
        arrow_source = args_dict.get("arrow_source", "sam")
        if arrow_source == "sam":
            print(f"[GPU {gpu_id}] Loading SAM3 (pretrained)...")
            sam_model = build_sam3_image_model(
                checkpoint_path="<MODEL_ROOT>/sam3/sam3.pt",
                device=device,
            )
            sam_model = sam_model.to(device)
            sam_model.eval()
            sam_processor = Sam3Processor(sam_model)
        elif arrow_source == "sam3_ft":
            ft_ckpt_path = args_dict["sam3_ft_checkpoint"]
            print(f"[GPU {gpu_id}] Loading SAM3 (fine-tuned: {ft_ckpt_path})...")
            sam_model = build_sam3_image_model(
                checkpoint_path=None,
                load_from_HF=False,
                device=device,
                eval_mode=False,
                enable_segmentation=True,
            )
            ft_ckpt = torch.load(ft_ckpt_path, map_location="cpu")
            sam_model.load_state_dict(ft_ckpt["model"], strict=False)
            sam_model = sam_model.to(device)
            sam_model.eval()
            sam_processor = Sam3Processor(sam_model)
        elif arrow_source == "yolo":
            yolo_ckpt = args_dict.get("yolo_checkpoint")
            assert yolo_ckpt and os.path.isfile(yolo_ckpt), \
                f"--yolo-checkpoint must exist: {yolo_ckpt}"
            print(f"[GPU {gpu_id}] Loading YOLO ({yolo_ckpt})...")
            from ultralytics import YOLO
            yolo_model = YOLO(yolo_ckpt)
        else:
            print(f"[GPU {gpu_id}] Arrow source: groundtruth, skipping detector.")

        # ── MiniCPM-V-4_5 VQA ──
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[GPU {gpu_id}] Loading MiniCPM tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args_dict["base_model_path"], trust_remote_code=True
        )
        print(f"[GPU {gpu_id}] Loading MiniCPM model (dtype={dtype})...")
        # when flash_attn is unavailable transformers falls back to sdpa; not forced here
        _model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        try:
            import flash_attn  # noqa: F401
            _model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass
        vqa_model = AutoModel.from_pretrained(
            args_dict["base_model_path"],
            **_model_kwargs,
        )
        vqa_model = vqa_model.to(device)

        if args_dict["adapter_path"]:
            print(f"[GPU {gpu_id}] Loading LoRA from {args_dict['adapter_path']}...")
            vqa_model = PeftModel.from_pretrained(
                vqa_model, args_dict["adapter_path"], device_map=device
            )
        else:
            print(f"[GPU {gpu_id}] Using base model without LoRA")
        vqa_model.eval()

        is_bpmn = args_dict.get("is_bpmn", False)
        is_flowgen = args_dict.get("is_flowgen", False)
        if is_flowgen:
            vqa_prompt = get_flowgen_triplet()
        elif is_bpmn:
            vqa_prompt = get_bpmn_triplet()
        else:
            vqa_prompt = get_triplet()
        # MiniCPM's tokenizer treats a literal "<image>" as the im_start_id special token,
        # would clash with the image placeholder chat() injects, misaligning start/end -> hstack crash.
        # strip the literal <image> tag hard-coded in the Qwen-style prompt.
        vqa_prompt = vqa_prompt.replace("<image>", "").lstrip()
        parse_fn = extract_and_parse_bpmn_output if (is_bpmn or is_flowgen) else extract_and_parse_vqa_output

        max_new_tokens = int(args_dict.get("max_new_tokens", 256))

        successful = 0
        failed = 0

        for img_idx, img_path in enumerate(image_list, 1):
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_dir = os.path.join(args_dict["output_dir"], base_name)
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, args_dict["result_filename"])

                if os.path.exists(result_path):
                    print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Skipping {base_name} (already exists)")
                    successful += 1
                    continue

                print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Processing {base_name}...")
                image_pil = Image.open(img_path).convert("RGB")

                # Arrow source
                if arrow_source == "groundtruth":
                    gt_arrows = load_gt_arrows(base_name, args_dict.get("image_dir", ""))
                    if not gt_arrows:
                        print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Warning: No groundtruth found for {base_name}")
                    arrow_iter = [
                        (arrow_idx, arr.get("matched_arrowhead_bbox"), arr.get("annotated_image_path"))
                        for arrow_idx, arr in enumerate(gt_arrows)
                    ]
                elif arrow_source == "yolo":
                    yolo_res = yolo_model.predict(
                        source=image_pil,
                        imgsz=args_dict.get("yolo_imgsz", 2048),
                        conf=args_dict.get("yolo_conf", 0.5),
                        device=device,
                        verbose=False,
                    )[0]
                    detected_boxes = []
                    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                        for xyxy in yolo_res.boxes.xyxy.cpu().tolist():
                            detected_boxes.append([int(x) for x in xyxy])
                    _ratio = args_dict.get("yolo_area_filter_median_ratio", 0.0)
                    if _ratio > 0 and len(detected_boxes) >= 3:
                        _areas = sorted((b[2]-b[0]) * (b[3]-b[1]) for b in detected_boxes)
                        _median = _areas[len(_areas)//2]
                        _thr = _median * _ratio
                        detected_boxes = [b for b in detected_boxes
                                          if (b[2]-b[0]) * (b[3]-b[1]) >= _thr]
                    if not detected_boxes:
                        print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Warning: YOLO detected no arrows in {base_name}")
                    arrow_iter = [(arrow_idx, bbox, None) for arrow_idx, bbox in enumerate(detected_boxes)]
                else:
                    state = sam_processor.set_image(image_pil)
                    arrowhead_output = sam_processor.set_text_prompt(state=state, prompt="arrowhead")
                    boxes = arrowhead_output["boxes"]
                    if boxes.numel() == 0:
                        arrowhead_output = sam_processor.set_text_prompt(state=state, prompt="arrow")
                        boxes = arrowhead_output["boxes"]
                    detected_boxes = []
                    for box in boxes:
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        detected_boxes.append([int(x) for x in box])
                    if not detected_boxes:
                        print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Warning: No arrows detected in {base_name}")
                    arrow_iter = [(arrow_idx, bbox, None) for arrow_idx, bbox in enumerate(detected_boxes)]

                all_triplets = []

                for arrow_idx, arrow_bbox, gt_img_path in arrow_iter:
                    arrow_label = f"H{arrow_idx + 1}"

                    # GT annotation strategy differs per dataset:
                    #   - flowgen: GT gives only bbox (same coord space as the IMAGE_DIR image), draw directly
                    #   - other datasets (flowvqa/fca/fcb/cbd/flowlearn): GT gives both bbox and
                    #     a pre-rendered annotated_image_path, whose bbox coord space may not match
                    #     matches the IMAGE_DIR source image (flowvqa measured a 4.6x offset); prefer the pre-rendered image
                    use_prerendered = (
                        arrow_source == "groundtruth"
                        and not is_flowgen
                        and gt_img_path is not None
                        and os.path.exists(gt_img_path)
                    )
                    if use_prerendered:
                        annotated_img = Image.open(gt_img_path).convert("RGB")
                    elif arrow_bbox is not None:
                        annotated_img = draw_bbox_and_tag_for_vqa(image_pil, arrow_bbox, arrow_label)
                    elif gt_img_path is not None:
                        annotated_img = Image.open(gt_img_path).convert("RGB")
                    else:
                        continue

                    if args_dict["save_debug"]:
                        debug_dir = os.path.join(output_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = os.path.join(debug_dir, f"arrow_{arrow_idx + 1}.png")
                        if gt_img_path is not None:
                            annotated_img.save(debug_path)
                        else:
                            save_debug_image(image_pil, arrow_bbox, debug_path)

                    # ── MiniCPM chat inference ──
                    msgs = [{"role": "user", "content": [annotated_img, vqa_prompt]}]
                    try:
                        with torch.no_grad():
                            vqa_output = vqa_model.chat(
                                image=None,
                                msgs=msgs,
                                tokenizer=tokenizer,
                                sampling=False,
                                max_new_tokens=max_new_tokens,
                            )
                        if isinstance(vqa_output, (list, tuple)):
                            vqa_output = vqa_output[0]
                    except Exception as e:
                        print(f"[GPU {gpu_id}] chat error on arrow {arrow_idx}: {e}")
                        vqa_output = ""

                    arrow_info = {
                        "arrow_id": arrow_idx + 1,
                        "arrow_label": arrow_label,
                        "arrow_bbox": arrow_bbox,
                    }
                    all_triplets.extend(parse_fn(vqa_output, arrow_info))

                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(all_triplets, f, indent=2, ensure_ascii=False)

                print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] ✓ Saved {len(all_triplets)} triplets to {result_path}")
                successful += 1
                torch.cuda.empty_cache()

            except Exception as e:
                failed += 1
                print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] ✗ Failed {img_path}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[GPU {gpu_id}] Batch Summary: {successful} successful, {failed} failed")

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker fatal init error: {e}")
        import traceback
        traceback.print_exc()

    print(f"[GPU {gpu_id}] Worker finished.")


# ============================================================
#   Main
# ============================================================

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    print(f"{'='*60}")
    print(f"MiniCPM-V-4_5 Arrow-per-image Pipeline")
    print(f"Backend: {args.backend.upper()}")
    print(f"{'='*60}")

    # Load the test set JSON (to filter files in image_dir)
    test_image_names = set()
    if args.test_json:
        print(f"Loading test set from: {args.test_json}")
        try:
            with open(args.test_json, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            if isinstance(test_data, dict):
                test_image_names = {Path(name).stem for name in test_data.keys()}
            elif isinstance(test_data, list):
                for item in test_data:
                    key = (
                        item.get("image")
                        or item.get("image_name")
                        or item.get("filename")
                        or item.get("file_name", "")
                    )
                    if key:
                        test_image_names.add(Path(key).stem)
            print(f"Test set contains {len(test_image_names)} images")
        except Exception as e:
            print(f"Warning: Failed to load test JSON: {e}")

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}")
    print(f"Base Model: {args.base_model_path}")
    print(f"Adapter: {args.adapter_path or 'None (base model only)'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Arrow Source: {args.arrow_source}")
    print(f"Output Dir: {args.output_dir}")

    image_dir_path = Path(args.image_dir)
    if not image_dir_path.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return

    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    all_image_files = []
    for ext in exts:
        all_image_files.extend(image_dir_path.glob(f"*{ext}"))

    if test_image_names:
        image_files = [str(p) for p in all_image_files if p.stem in test_image_names]
    else:
        image_files = [str(p) for p in all_image_files]
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {args.image_dir}")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process.\n")

    is_flowgen = args.is_flowgen or "flowgen" in args.image_dir.lower()
    is_bpmn = args.is_bpmn or (not is_flowgen and "bpmn" in args.image_dir.lower())
    if is_flowgen:
        print("Using FlowGen 5-field output format (group/container).")
    elif is_bpmn:
        print("Using BPMN 5-field output format (pool/lane).")

    args_dict = {
        "output_dir": args.output_dir,
        "result_filename": args.result_filename,
        "save_debug": args.save_debug,
        "adapter_path": args.adapter_path,
        "base_model_path": args.base_model_path,
        "arrow_source": args.arrow_source,
        "sam3_ft_checkpoint": args.sam3_ft_checkpoint,
        "yolo_checkpoint": args.yolo_checkpoint,
        "yolo_conf": args.yolo_conf,
        "yolo_imgsz": args.yolo_imgsz,
        "yolo_area_filter_median_ratio": args.yolo_area_filter_median_ratio,
        "is_bpmn": is_bpmn,
        "is_flowgen": is_flowgen,
        "image_dir": args.image_dir,
        "max_new_tokens": args.max_new_tokens,
    }

    num_batches = math.ceil(total_images / args.batch_size)
    for b_idx in range(num_batches):
        start_idx = b_idx * args.batch_size
        end_idx = min((b_idx + 1) * args.batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]

        print(f"\n{'#'*60}")
        print(f"Processing Batch {b_idx+1}/{num_batches} (Images {start_idx+1} - {end_idx})")
        print(f"{'#'*60}")

        wpg = args.workers_per_gpu
        worker_slots = [(gid, s) for gid in gpu_ids for s in range(wpg)]
        chunks = {slot: [] for slot in worker_slots}
        for i, img in enumerate(batch_files):
            chunks[worker_slots[i % len(worker_slots)]].append(img)

        processes = []
        for slot in worker_slots:
            if not chunks[slot]:
                continue
            gid, _ = slot
            p = mp.Process(
                target=creation_worker_local,
                args=(gid, chunks[slot], args_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f">>> Batch {b_idx+1}/{num_batches} Completed.")

    print(f"\n{'='*60}")
    print(f"✔ All Batches Completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
