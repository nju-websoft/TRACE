import os
import json
import argparse
import re
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageDraw
from lora.prompt import get_triplet, get_bpmn_triplet, get_flowgen_triplet
import math

def repair_and_load_json(text: str):
    from json_repair import repair_json, loads
    try:
        return loads(repair_json(text, ensure_ascii=False))
    except Exception as e:
        print(f"JSON parse failed: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="<OUTPUT_ROOT>/output_lora",
                        help="Output directory")
    parser.add_argument("--image-dir", type=str, 
                        default="<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images",
                        help="Source image directory")
    parser.add_argument("--test-json", type=str,
                        default="",
                        help="Path to the test set JSON")
    parser.add_argument("--result-filename", type=str, default="arrow_triplets.json",
                        help="Arrow result filename")
    parser.add_argument("--save-debug", action="store_true",
                        help="Whether to save debug images")
    parser.add_argument("--adapter-path", type=str,
                        default=None,
                        help="Path to the LoRA adapter; omit to use the base model only")
    parser.add_argument("--base-model-path", type=str, default="<MODEL_ROOT>/Qwen2.5-VL-3B-Instruct",
                        help="Path to the Qwen2-VL base model")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="GPU IDs")
    parser.add_argument("--batch-size", type=int, default=800,
                        help="Images per batch")
    parser.add_argument("--backend", type=str, default="local", choices=["local"],
                        help="Inference backend: local")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Workers to launch per GPU (2 is fine for small 3B/4B models)")
    parser.add_argument("--arrow-source", type=str, default="sam",
                        choices=["sam", "sam3_ft", "yolo", "groundtruth"],
                        help="Arrow source: sam / sam3_ft / yolo (YOLO11x) / groundtruth")
    parser.add_argument("--sam3-ft-checkpoint", type=str,
                        default="<REPO_ROOT>/detection/runs/arrowhead_detection/checkpoints/checkpoint_10.pt",
                        help="Path to the fine-tuned SAM3 checkpoint (used when arrow-source=sam3_ft)")
    parser.add_argument("--yolo-checkpoint", type=str, default=None,
                        help="Path to YOLO best.pt (used when arrow-source=yolo)")
    parser.add_argument("--yolo-conf", type=float, default=0.5,
                        help="YOLO inference confidence threshold (default 0.5)")
    parser.add_argument("--yolo-imgsz", type=int, default=2048,
                        help="YOLO inference image size (default 2048, matching training)")
    parser.add_argument("--yolo-area-filter-median-ratio", type=float, default=0.0,
                        help="Post-process: drop bboxes with area < median*ratio (0 = no filter; 0.6 recommended for flowgen_hard)")
    parser.add_argument("--is-bpmn", action="store_true",
                        help="Use the BPMN 5-field output format (pool/lane)")
    parser.add_argument("--is-flowgen", action="store_true",
                        help="Use the FlowGen 5-field output format (group/container)")
    return parser.parse_args()

def extract_and_parse_vqa_output(vqa_output: str, arrow_info: Dict) -> List[Dict]:
    """
    Parse the VQA output, handling the '||' separator and None values.
    
    Args:
        vqa_output: the LLM's raw output string
        arrow_info: dict of metadata (arrow_id, arrow_label, arrow_bbox, ...)
        
    Returns:
        List[Dict]: the parsed list of triplet dicts
    """
    # 1. Remove <think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', vqa_output, flags=re.DOTALL)
    cleaned = cleaned.strip()
    
    # 2. Regex-extract raw strings
    source_match = re.search(r'source_nodes:\s*(.+?)(?=\n\s*(?:end_nodes:|condition:)|$)', cleaned, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r'end_nodes:\s*(.+?)(?=\n\s*(?:condition:)|$)', cleaned, re.IGNORECASE | re.DOTALL)
    condition_match = re.search(r'condition:\s*(.+?)(?=$)', cleaned, re.IGNORECASE | re.DOTALL)
    
    raw_source = source_match.group(1).strip() if source_match else "[UNKNOWN]"
    raw_end = end_match.group(1).strip() if end_match else "[UNKNOWN]"
    raw_condition = condition_match.group(1).strip() if condition_match else "None"
    
    # 3. Split on '||' and strip whitespace
    sources = [s.strip() for s in raw_source.split('||')]
    ends = [e.strip() for e in raw_end.split('||')]
    conditions = [c.strip() for c in raw_condition.split('||')]
    
    # 4. Alignment (broadcasting)
    # if one field has a single value while others have several (e.g. 1 source, 2 ends), broadcast it
    max_len = max(len(sources), len(ends), len(conditions))
    
    if len(sources) == 1 and max_len > 1:
        sources = sources * max_len
    if len(ends) == 1 and max_len > 1:
        ends = ends * max_len
    if len(conditions) == 1 and max_len > 1:
        conditions = conditions * max_len
        
    # if still misaligned (e.g. sources=2, ends=3), use min_len to avoid errors (or max_len with padding)
    # for robustness, use max_len and pad with the last value
    final_triplets = []
    
    for i in range(max_len):
        s_text = sources[i] if i < len(sources) else sources[-1]
        e_text = ends[i] if i < len(ends) else ends[-1]
        c_text = conditions[i] if i < len(conditions) else "None"
        
        # 5. Handle None values
        # if the string is "None" (case-insensitive), store null (Python None) in the JSON
        final_condition = None if c_text.lower() == "none" else c_text
        
        triplet = {
            "arrow_id": arrow_info["arrow_id"],
            "arrow_label": arrow_info["arrow_label"],
            "source": s_text,
            "end": e_text,
            "condition": final_condition, # if None, json.dump writes null
            "arrow_bbox": arrow_info["arrow_bbox"],
            "raw_vqa_output": vqa_output if i == 0 else "" # keep raw output only on the first item to save space (or keep all)
        }
        final_triplets.append(triplet)
        
    return final_triplets

def extract_and_parse_bpmn_output(vqa_output: str, arrow_info: Dict) -> List[Dict]:
    """
    Parse the BPMN VQA output (5 fields).
    Output format:
        source_node: ...
        source_location: ...
        condition: ...
        target_node: ...
        target_location: ...
    """
    cleaned = re.sub(r'<think>.*?</think>', '', vqa_output, flags=re.DOTALL).strip()

    def extract_field(name):
        m = re.search(rf'{name}:\s*(.+?)(?:\n|$)', cleaned, re.IGNORECASE)
        return m.group(1).strip() if m else "None"

    source_node = extract_field("source_node")
    source_location = extract_field("source_location")
    condition = extract_field("condition")
    target_node = extract_field("target_node")
    target_location = extract_field("target_location")

    # Normalize "None" string
    if condition.lower() == "none":
        condition = "None"
    if source_location.lower() == "none":
        source_location = "None"
    if target_location.lower() == "none":
        target_location = "None"

    triplet = {
        "arrow_id": arrow_info["arrow_id"],
        "arrow_label": arrow_info["arrow_label"],
        "source_node": source_node,
        "source_location": source_location,
        "condition": condition,
        "target_node": target_node,
        "target_location": target_location,
        "arrow_bbox": arrow_info["arrow_bbox"],
        "raw_vqa_output": vqa_output,
    }
    return [triplet]


def draw_bbox_and_tag_for_vqa(image_pil: Image.Image, box: list, text_label: str) -> Image.Image:
    """Draw a blue bounding box on the image."""
    img_copy = image_pil.copy()
    draw = ImageDraw.Draw(img_copy)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
    return img_copy

def save_debug_image(image_pil: Image.Image, bbox: list, output_path: str):
    """Save a debug image with the bounding box drawn."""
    try:
        img_with_box = draw_bbox_and_tag_for_vqa(image_pil, bbox, "DEBUG")
        img_with_box.save(output_path)
    except Exception as e:
        print(f"Warning: Failed to save debug image {output_path}: {e}")

def image_to_base64(image_pil: Image.Image) -> str:
    """Convert a PIL image to a base64 string."""
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


_GT_BASE_DIR = "<DATA_ROOT>/data_4_training"

def load_gt_arrows(image_stem: str, image_dir: str = "") -> list:
    """
    Load pre-annotated arrow info from the groundtruth directory.
    Scan all *_test* subdirs under _GT_BASE_DIR to find the json matching image_stem.

    Args:
        image_stem: image filename without extension, e.g. "diagrams_101"
        image_dir: image directory, used to infer flowgen difficulty and avoid cross-difficulty mismatches
                   e.g. ".../test_img_medium" -> search only within flowgen_test_medium

    Returns:
        arrows: list of arrows; each has matched_arrowhead_bbox (coords in the source image)
    """
    base = Path(_GT_BASE_DIR)

    # Infer flowgen difficulty from image_dir: test_img_easy / test_img_medium / test_img_hard
    flowgen_diff = ""
    if image_dir:
        dir_name = Path(image_dir).name  # e.g. "test_img_medium" or "test_img_medium_orig"
        for d in ("easy", "medium", "hard"):
            if d in dir_name:
                flowgen_diff = d
                break

    # scan all subdirs containing "_test"
    for test_dir in sorted(base.glob("*_test*")):
        if not test_dir.is_dir():
            continue
        # direct match (fcb_test, bpmn_test, etc.)
        json_path = test_dir / image_stem / f"{image_stem}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("arrows", [])
        # flowgen format: image_stem="diagrams_101", dir name="diagrams_easy_diagrams_101"
        if test_dir.name.startswith("flowgen_test_"):
            # if the difficulty is known, only search the matching dir
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
#   Worker: SAM3 + VQA with Local Model
# ============================================================

def creation_worker_local(gpu_id: int, image_list: List[str], args_dict: Dict):
    """
    Local-model worker: SAM3 detection + local VQA model
    """
    if not image_list:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    # Generic HF VLM version (Gemma / LLaVa-Next, etc.): no longer uses qwen_vl_utils.process_vision_info,
    # feed the PIL.Image directly to AutoProcessor.
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    
    device = "cuda:0"
    
    print(f"[GPU {gpu_id}] Local worker started. Tasks: {len(image_list)}")
    
    try:
        # Load the detector (skipped in groundtruth mode)
        sam_processor = None
        yolo_model = None
        arrow_source = args_dict.get("arrow_source", "sam")
        if arrow_source == "sam":
            print(f"[GPU {gpu_id}] Loading SAM3 (pretrained)...")
            sam_model = build_sam3_image_model(
                checkpoint_path="<MODEL_ROOT>/sam3/sam3.pt",
                device=device
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

        # Load the VQA model
        print(f"[GPU {gpu_id}] Loading VQA...")
        vqa_model = AutoModelForImageTextToText.from_pretrained(
            args_dict["base_model_path"],
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        if args_dict["adapter_path"]:
            print(f"[GPU {gpu_id}] Loading LoRA from {args_dict['adapter_path']}...")
            vqa_model = PeftModel.from_pretrained(vqa_model, args_dict["adapter_path"])
        else:
            print(f"[GPU {gpu_id}] Using base model without LoRA")
        vqa_model.eval()
        vqa_processor = AutoProcessor.from_pretrained(args_dict["base_model_path"])

        is_bpmn = args_dict.get("is_bpmn", False)
        is_flowgen = args_dict.get("is_flowgen", False)
        if is_flowgen:
            vqa_prompt = get_flowgen_triplet()
        elif is_bpmn:
            vqa_prompt = get_bpmn_triplet()
        else:
            vqa_prompt = get_triplet()
        # the literal <image> in the template is for Qwen-VL; generic HF VLMs via apply_chat_template
        # (LLaVa-Next / Gemma3 / etc.) insert another <image> themselves, so strip it here first.
        vqa_prompt = vqa_prompt.replace("<image>", "").lstrip()
        parse_fn = extract_and_parse_bpmn_output if (is_bpmn or is_flowgen) else extract_and_parse_vqa_output

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

                # Arrow source: SAM3 / YOLO detection, or groundtruth pre-annotation
                if args_dict.get("arrow_source", "sam") == "groundtruth":
                    gt_arrows = load_gt_arrows(base_name, args_dict.get("image_dir", ""))
                    if not gt_arrows:
                        print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Warning: No groundtruth found for {base_name}")
                    arrow_iter = [
                        (arrow_idx, arr.get("matched_arrowhead_bbox"), arr.get("annotated_image_path"))
                        for arrow_idx, arr in enumerate(gt_arrows)
                    ]
                elif args_dict.get("arrow_source") == "yolo":
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
                        boxes = arrowhead_output['boxes']
                    detected_boxes = []
                    for box in boxes:
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        detected_boxes.append([int(x) for x in box])
                    if not detected_boxes:
                        print(f"[GPU {gpu_id}] [{img_idx}/{len(image_list)}] Warning: No arrows detected in {base_name}")
                    arrow_iter = [(arrow_idx, bbox, None) for arrow_idx, bbox in enumerate(detected_boxes)]

                # Identify each arrow
                all_triplets = []

                for arrow_idx, arrow_bbox, gt_img_path in arrow_iter:
                    arrow_label = f"H{arrow_idx + 1}"

                    # flowgen GT bbox shares the coordinate space of the IMAGE_DIR source image -> draw on it
                    # other datasets' GT provides a pre-rendered image; bbox coord space may differ from the source
                    # (flowvqa measured a 4.6x offset) -> prefer the pre-rendered image
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
                    
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": vqa_prompt},
                        ],
                    }]

                    text_input = vqa_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # Generic HF VLM: pass the PIL.Image list straight to the processor
                    inputs = vqa_processor(
                        text=[text_input],
                        images=[annotated_img],
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    
                    with torch.no_grad():
                        generated_ids = vqa_model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            temperature=0.1
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    vqa_output = vqa_processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    # Build the base info
                    arrow_info = {
                        "arrow_id": arrow_idx + 1,
                        "arrow_label": arrow_label,
                        "arrow_bbox": arrow_bbox
                    }
                    
                    # Parse and split the result (may return multiple triplets)
                    parsed_triplets = parse_fn(vqa_output, arrow_info)
                    all_triplets.extend(parsed_triplets)

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
#   Main Pipeline
# ============================================================

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    
    print(f"{'='*60}")
    print(f"Batched Flowchart Analysis Pipeline")
    print(f"{'='*60}")
    
    # Load the test setJSON
    test_image_names = set()
    if args.test_json:
        print(f"Loading test set from: {args.test_json}")
        try:
            with open(args.test_json, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            test_image_names = {Path(name).stem for name in test_data.keys()}
            print(f"Test set contains {len(test_image_names)} images")
        except Exception as e:
            print(f"Warning: Failed to load test JSON: {e}")
            print("Processing all images in directory...")
    
    # Parse GPUs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}")
    print(f"Base Model: {args.base_model_path}")
    print(f"Adapter: {args.adapter_path or 'None (base model only)'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Arrow Source: {args.arrow_source}")
    print(f"Output Dir: {args.output_dir}")
    
    # Gather all images and filter by the test set
    image_dir_path = Path(args.image_dir)
    if not image_dir_path.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    all_image_files = []
    for ext in exts:
        all_image_files.extend(image_dir_path.glob(f"*{ext}"))
    
    if test_image_names:
        image_files = [
            str(img_path) for img_path in all_image_files 
            if img_path.stem in test_image_names
        ]
    else:
        image_files = [str(img_path) for img_path in all_image_files]
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        if test_image_names:
            print(f"(Looking for test set images: {list(test_image_names)[:5]}...)")
        return
    
    total_images = len(image_files)
    print(f"Found {total_images} images to process.\n")
    
    # Decide whether to use the 5-field format (CLI arg wins; otherwise inferred from the path)
    is_flowgen = args.is_flowgen or "flowgen" in args.image_dir.lower()
    is_bpmn = args.is_bpmn or (not is_flowgen and "bpmn" in args.image_dir.lower())
    if is_flowgen:
        print(f"Using FlowGen 5-field output format (group/container).")
    elif is_bpmn:
        print(f"Using BPMN 5-field output format (pool/lane).")

    # Build the kwargs dict
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
    }

    # Pick the worker function
    worker_func = creation_worker_local

    # process in batches
    num_batches = math.ceil(total_images / args.batch_size)
    
    for b_idx in range(num_batches):
        start_idx = b_idx * args.batch_size
        end_idx = min((b_idx + 1) * args.batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"\n{'#'*60}")
        print(f"Processing Batch {b_idx+1}/{num_batches} (Images {start_idx+1} - {end_idx})")
        print(f"{'#'*60}")
        
        # split the current batch evenly across worker slots
        # each GPU can run multiple workers (controlled by workers_per_gpu)
        wpg = args.workers_per_gpu
        worker_slots = []  # (gpu_id, slot_index)
        for gid in gpu_ids:
            for s in range(wpg):
                worker_slots.append((gid, s))

        creation_chunks = {slot: [] for slot in worker_slots}
        for i, img in enumerate(batch_files):
            creation_chunks[worker_slots[i % len(worker_slots)]].append(img)

        # Start processes
        creation_processes = []
        for slot in worker_slots:
            if not creation_chunks[slot]:
                continue
            gid, slot_idx = slot
            p = mp.Process(
                target=worker_func,
                args=(gid, creation_chunks[slot], args_dict)
            )
            p.start()
            creation_processes.append(p)
        
        # Wait for all processes to finish
        for p in creation_processes:
            p.join()
        
        print(f">>> Batch {b_idx+1}/{num_batches} Completed.")
    
    print(f"\n{'='*60}")
    print(f"✔ All Batches Completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()