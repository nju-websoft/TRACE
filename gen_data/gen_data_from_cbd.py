import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import math
from difflib import SequenceMatcher
import torch
from PIL import Image
import re
import json
from pathlib import Path

# === Libraries ===
from paddleocr import PaddleOCR

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Warning: SAM3 not found.")

def get_center(bbox):
    """
    Compute the center coordinates of a box
    Input: bbox = [xmin, ymin, xmax, ymax]
    Output: (center_x, center_y)
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (cx, cy)

def get_distance(pt1, pt2):
    """
    Compute the Euclidean distance between two points
    Input: pt1 = (x1, y1), pt2 = (x2, y2)
    Output: float distance
    """
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def get_box_distance(point, bbox):
    """
    Compute the shortest distance from a point to a box
    Input: point = (x, y), bbox = [xmin, ymin, xmax, ymax]
    Output: float distance (0 if point is inside the box)
    """
    px, py = point
    xmin, ymin, xmax, ymax = bbox
    
    # If point is inside the box, distance is 0
    if xmin <= px <= xmax and ymin <= py <= ymax:
        return 0.0
    
    # Calculate distance to the nearest edge/corner
    dx = max(xmin - px, 0, px - xmax)
    dy = max(ymin - py, 0, py - ymax)
    
    return math.sqrt(dx**2 + dy**2)

def bbox_iou(bbox1, bbox2):
    """
    Compute the IoU (Intersection over Union) of two boxes
    Input: bbox1 = [xmin, ymin, xmax, ymax], bbox2 = [xmin, ymin, xmax, ymax]
    Output: float IoU (0 to 1)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    if intersection == 0:
        return 0.0
    
    # Calculate union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = bbox1_area + bbox2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def expand_bbox(bbox, margin):
    """
    Expand a bbox
    Input: bbox = [xmin, ymin, xmax, ymax], margin = pixels to expand
    Output: expanded bbox
    """
    return [
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin
    ]

def bbox_overlap_ratio(bbox1, bbox2):
    """
    Compute the overlap ratio of two boxes (intersection area / bbox1 area)
    Input: bbox1 = [xmin, ymin, xmax, ymax], bbox2 = [xmin, ymin, xmax, ymax]
    Output: float ratio (0 to 1)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate bbox1 area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    if bbox1_area == 0:
        return 0.0
    
    return intersection / bbox1_area

# ==========================================
# 1. Standard Text Matcher (New Module)
# ==========================================

class StandardTextMatcher:
    def __init__(self, gt_txt_path):
        self.gt_data = self._load_and_parse_gt(gt_txt_path)
        
    def _load_and_parse_gt(self, path):
        """
        Parses the specific format and extracts only the structured part before ||
        Returns a list of dicts, where each dict contains:
        - 'node_vocab': set of node names (from <H> and <T>), with underscores replaced by spaces
        - 'relation_vocab': set of relation names (from <R>), excluding 'connected_with'
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        dataset_vocab = []  # List of dicts
        
        for line in lines:
            line = line.strip()
            if not line: 
                continue
            
            # Split by || to get only the structured part
            if '||' in line:
                structured_part = line.split('||')[0].strip()
            else:
                structured_part = line
            
            # Extract node texts: everything after <H> until <R>
            heads = re.findall(r'<H>\s*(.*?)\s*<R>', structured_part)
            # Extract node texts: everything after <T> until <H> or End of Line
            tails = re.findall(r'<T>\s*(.*?)\s*(?:<H>|$)', structured_part)
            
            # Extract relation texts: everything after <R> until <T>
            relations = re.findall(r'<R>\s*(.*?)\s*<T>', structured_part)
            
            # Replace underscores with spaces in nodes
            node_vocab = set(node.replace('_', ' ') for node in heads + tails)
            
            # Filter out 'connected_with' and replace underscores with spaces in relations
            relation_vocab = set(
                rel.replace('_', ' ') for rel in relations 
                if rel != 'connected_with'
            )
            dataset_vocab.append({
                'node_vocab': node_vocab,
                'relation_vocab': relation_vocab
            })
            
        return dataset_vocab

    def get_vocabulary_for_image(self, image_index):
        """
        Returns the node and relation vocabularies for a specific image index.
        Returns: dict with 'node_vocab' and 'relation_vocab' sets
        """
        if 0 <= image_index < len(self.gt_data):
            return self.gt_data[image_index]
        return {'node_vocab': set(), 'relation_vocab': set()}

    def find_best_match(self, query_text, vocabulary, type='node'):
        """
        Fuzzy match query_text against the vocabulary set.
        Returns the best match from vocabulary.
        """
        if not query_text or not vocabulary:
            return query_text # Fallback to original if no vocab or empty query
        
        if type == 'node':
            vocabulary = vocabulary['node_vocab']
        else:
            vocabulary = vocabulary['relation_vocab']
        best_match = None
        highest_ratio = 0.0
        
        for standard_text in vocabulary:
            # Normalize for comparison (ignore case, spacing differences)
            std_norm = standard_text.replace('_', ' ').lower()
            qry_norm = query_text.replace('_', ' ').lower()
            
            ratio = SequenceMatcher(None, qry_norm, std_norm).ratio()
            
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = standard_text
        
        if highest_ratio > 0.3: 
            return best_match
        return query_text 

# ==========================================
# 2. Main Recogniser Class
# ==========================================

class FlowchartRecogniser:
    def __init__(self, sam_checkpoint_path, gt_txt_path=None):
        # 1. Init SAM3
        print("Loading SAM3 Model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.sam_model = build_sam3_image_model(checkpoint_path=sam_checkpoint_path, device=self.device)
            self.sam_processor = Sam3Processor(self.sam_model)
        except:
            self.sam_model = None
            print("SAM3 Model failed (Mocking).")

        # 2. Init OCR
        print("Loading OCR Model...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # 3. Init Text Matcher
        self.matcher = None
        if gt_txt_path:
            self.matcher = StandardTextMatcher(gt_txt_path)
            print(f"Loaded Standard Text Matcher with {len(self.matcher.gt_data)} lines.")

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = {'arrow': [], 'node': [], 'text': [], 'node_text': [], 'condition_text': []}
        
        # First pass: collect all objects
        for obj in root.findall('object'):
            name = obj.find('name').text
            bnd = obj.find('bndbox')
            bbox = [int(bnd.find('xmin').text), int(bnd.find('ymin').text),
                    int(bnd.find('xmax').text), int(bnd.find('ymax').text)]
            item = {'name': name, 'bbox': bbox, 'center': get_center(bbox)}
            
            if name == 'arrow': 
                objects['arrow'].append(item)
            elif name == 'text': 
                objects['text'].append(item)
            else: 
                objects['node'].append(item)
        
        # Second pass: classify text as node_text or condition_text
        for text_item in objects['text']:
            text_bbox = text_item['bbox']
            is_inside_node = False
            
            # Check if this text is inside any node
            for node in objects['node']:
                node_bbox = node['bbox']
                # Check if text_bbox is inside node_bbox
                if (text_bbox[0] >= node_bbox[0] and text_bbox[1] >= node_bbox[1] and
                    text_bbox[2] <= node_bbox[2] and text_bbox[3] <= node_bbox[3]):
                    is_inside_node = True
                    objects['node_text'].append(text_item)
                    break
            
            if not is_inside_node:
                objects['condition_text'].append(text_item)
        
        return objects

    def get_arrowheads_from_sam3(self, image_path):
        if self.sam_model is None: return []
        image_pil = Image.open(image_path).convert("RGB")
        state = self.sam_processor.set_image(image_pil)
        output = self.sam_processor.set_text_prompt(state=state, prompt="arrowhead")
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        arrowheads = []
        for i, score in enumerate(scores):
            if score > 0.25:
                box = boxes[i].int().tolist()
                arrowheads.append({'bbox': box, 'center': get_center(box), 'score': float(score)})
        return arrowheads

    def recognize_text(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return ""
        res = self.ocr.predict(crop)
        return " ".join(res[0]['rec_texts'])

    def process_image(self, image_path, xml_path, image_index, output_dir, debug=False):
        """
        Process a single image and return its arrow data
        Args:
            debug: if True, draw rectangles for all arrows on the image
        """
        print(f"\nProcessing: {image_path}")
        
        gt_objects = self.parse_xml(xml_path)
        img_cv2 = cv2.imread(image_path)
        
        # Get Vocabulary for this specific image
        current_vocab = set()
        if self.matcher:
            current_vocab = self.matcher.get_vocabulary_for_image(image_index)

        detected_arrowheads = self.get_arrowheads_from_sam3(image_path)
        image_name = Path(image_path).stem
        
        # Create image-specific folder structure
        image_folder = output_dir / image_name
        image_folder.mkdir(parents=True, exist_ok=True)
        
        annotated_images_folder = image_folder / "annotated_images"
        annotated_images_folder.mkdir(parents=True, exist_ok=True)
        
        # Debug: create an image with all arrow boxes
        if debug:
            debug_img = img_cv2.copy()
            for arrow in gt_objects['arrow']:
                arrow_box = arrow['bbox']
                cv2.rectangle(debug_img, 
                            (arrow_box[0], arrow_box[1]), 
                            (arrow_box[2], arrow_box[3]), 
                            (0, 255, 0),  # Green color in BGR
                            2)  # Line width
            debug_path = image_folder / f"{image_name}_debug_arrows.png"
            cv2.imwrite(str(debug_path), debug_img)
            print(f"Saved debug image with all arrows: {debug_path}")
        
        # Step 1: recognize all condition_text and find the best match
        condition_to_arrow_map = {}  # {condition_index: (arrow_index, iou)}
        
        for cond_idx, cond_text in enumerate(gt_objects['condition_text']):
            # expand by 5px and recognize text
            expanded_cond_box = expand_bbox(cond_text['bbox'], 5)
            cond_content = self.recognize_text(img_cv2, expanded_cond_box)
            
            # find the most similar canonical text
            cond_content = self.matcher.find_best_match(cond_content, current_vocab, "relation") if self.matcher else cond_content
            
            if not cond_content or not cond_content.strip():
                continue
            
            # iterate all arrows, find the one with the highest IoU
            best_arrow_idx = None
            best_iou = 0.0
            
            for arrow_idx, arrow in enumerate(gt_objects['arrow']):
                # expand the arrow by 15px
                expanded_arrow_box = expand_bbox(arrow['bbox'], 20)
                
                # compute IoU
                iou = bbox_iou(cond_text['bbox'], expanded_arrow_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_arrow_idx = arrow_idx
            
            # keep only those that intersect (iou > 0)
            if best_iou > 0 and best_arrow_idx is not None:
                condition_to_arrow_map[cond_idx] = {
                    'arrow_idx': best_arrow_idx,
                    'iou': best_iou,
                    'text': cond_content.strip()
                }
        
        # Step 2: process each arrow
        arrows_data = []
        arrow_id_counter = 1
        
        for arrow_idx, arrow in enumerate(gt_objects['arrow']):
            # 1. Match Head (Start/End determination) - skip if no match
            arrow_box = arrow['bbox']
            matched_head = None
            # Find best matching arrowhead inside arrow bbox
            for head in detected_arrowheads:
                hx, hy = head['center']
                if arrow_box[0] <= hx <= arrow_box[2] and arrow_box[1] <= hy <= arrow_box[3]:
                    matched_head = head
                    break 
            
            # if no arrowhead matched, skip this arrow
            if matched_head is None:
                print(f"Arrow {arrow_idx} discarded: No arrowhead matched")
                continue
            
            # Create individual annotated image for this arrow
            annotated_img = img_cv2.copy()
            
            # Determine geometry
            head_pt = matched_head['center']
            corners = [(arrow_box[0], arrow_box[1]), (arrow_box[2], arrow_box[1]),
                    (arrow_box[0], arrow_box[3]), (arrow_box[2], arrow_box[3])]
            tail_pt = max(corners, key=lambda c: get_distance(c, head_pt))
            
            # expand the arrowhead bbox by 5px, draw a blue rectangle, width 3
            head_bbox = matched_head['bbox']
            expanded_head_bbox = expand_bbox(head_bbox, 5)
            cv2.rectangle(annotated_img, 
                        (expanded_head_bbox[0], expanded_head_bbox[1]), 
                        (expanded_head_bbox[2], expanded_head_bbox[3]), 
                        (255, 0, 0),  # Blue color in BGR
                        3)  # Line width

            # 2. Find Nodes
            source_node = min(gt_objects['node'], key=lambda n: get_box_distance(tail_pt, n['bbox']))
            target_node = min(gt_objects['node'], key=lambda n: get_box_distance(head_pt, n['bbox']))
            
            # 3. OCR & Correction for nodes
            raw_src = self.recognize_text(img_cv2, source_node['bbox'])
            raw_tgt = self.recognize_text(img_cv2, target_node['bbox'])
            std_src = self.matcher.find_best_match(raw_src, current_vocab) if self.matcher else raw_src
            std_tgt = self.matcher.find_best_match(raw_tgt, current_vocab) if self.matcher else raw_tgt
            
            # 4. Filter out arrows whose start == end
            if std_src == std_tgt:
                print(f"Arrow {arrow_idx} discarded: Start and end are the same ({std_src})")
                continue
            
            # 5. Find the conditions belonging to this arrow
            condition_texts = []
            for cond_idx, cond_info in condition_to_arrow_map.items():
                if cond_info['arrow_idx'] == arrow_idx:
                    condition_texts.append({
                        'text': cond_info['text'],
                        'iou': cond_info['iou']
                    })
            
            # Sort by IoU (highest first) and extract text
            condition_texts.sort(key=lambda x: x['iou'], reverse=True)
            print(f"Arrow {arrow_idx} conditions: {condition_texts}")
            
            # keep only the condition with the highest IoU (if any)
            if len(condition_texts) >= 1:
                final_conditions = [condition_texts[0]['text']]
            else:
                final_conditions = []
            
            # Save individual annotated image with arrow_id in filename
            output_image_path = annotated_images_folder / f"{image_name}_arrow_{arrow_id_counter}.png"
            cv2.imwrite(str(output_image_path), annotated_img)
            print(f"Saved annotated image: {output_image_path}")
            
            arrow_data = {
                "arrow_id": arrow_id_counter,
                "start_point": [std_src],
                "end_point": [std_tgt],
                "condition": final_conditions,
                "matched_arrowhead_bbox": matched_head['bbox'],
                "annotated_image_path": str(output_image_path)
            }
            
            arrows_data.append(arrow_data)
            arrow_id_counter += 1

        return arrows_data, image_folder

# ==========================================
# 3. Batch Processing
# ==========================================

def natural_sort_key(path):
    """
    Natural-sort key function; works for filenames with various prefixes
    e.g. Break (1).png, Connect (2).png, Test (10).png, etc.
    """
    import re
    text = str(path.stem)
    # split the string into text and numeric parts
    # e.g. "Break (123)" -> ['Break (', '123', ')']
    parts = re.split(r'(\d+)', text)
    # numeric parts to int, text parts to lowercase
    return [int(part) if part.isdigit() else part.lower() for part in parts]

def process_all_images(sam_ckpt, train_txt_path, train_img_dir, train_xml_dir, output_dir, debug=False):
    """
    Batch-process all training images
    Args:
        debug: if True, generate a debug image per image showing all arrow boxes
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize recognizer
    recognizer = FlowchartRecogniser(sam_ckpt, gt_txt_path=train_txt_path)
    
    # Get all image files and sort naturally
    train_img_dir = Path(train_img_dir)
    image_files = sorted(list(train_img_dir.glob("*.png")), key=natural_sort_key)
    
    print(f"Found {len(image_files)} images to process")
    print(f"First few images in order: {[img.name for img in image_files[:5]]}")
    if debug:
        print("Debug mode enabled: Will generate debug images showing all arrow boxes")
    
    # Process each image
    for idx, img_path in enumerate(image_files):
        try:
            # Get corresponding XML file
            img_name = img_path.stem
            xml_path = Path(train_xml_dir) / f"{img_name}.xml"
            
            if not xml_path.exists():
                print(f"Warning: XML file not found for {img_name}, skipping...")
                continue
            
            # Process image
            arrows_data, image_folder = recognizer.process_image(
                str(img_path), 
                str(xml_path), 
                image_index=idx,
                output_dir=output_dir,
                debug=debug
            )
            
            # Save JSON file in the image-specific folder
            json_output_path = image_folder / f"{img_name}.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "image_name": img_name,
                    "image_path": str(img_path),
                    "arrows": arrows_data
                }, f, indent=2, ensure_ascii=False)
            
            print(f"Saved JSON: {json_output_path}")
            print(f"Progress: {idx+1}/{len(image_files)}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    # Configuration
    sam_ckpt = "<MODEL_ROOT>/sam3/sam3.pt"
    train_txt_path = "<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/val.txt"
    train_img_dir = "<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/val_img"
    train_xml_dir = "<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/xml_files/val"
    output_dir = "<DATA_ROOT>/data_4_training/cbd"
    
    # Set debug=True to generate debug images showing all arrow boxes
    DEBUG_MODE = False
    
    # Run batch processing
    process_all_images(sam_ckpt, train_txt_path, train_img_dir, train_xml_dir, output_dir, debug=DEBUG_MODE)
    
    print("\n=== Processing Complete ===")