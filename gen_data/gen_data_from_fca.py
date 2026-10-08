import xml.etree.ElementTree as ET
import numpy as np
import json
import os
import cv2
import math
import torch
from pathlib import Path
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Warning: SAM3 not found.")

# ================= Configuration =================
MAX_DIMENSION = 1333
PADDING = 20
LINE_THICKNESS = 2
# ===========================================

def get_center(bbox):
    """Compute the center coordinates of a box."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (cx, cy)

def get_distance(pt1, pt2):
    """Compute the Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def bbox_iou(bbox1, bbox2):
    """Compute the IoU of two boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    if intersection == 0:
        return 0.0
    
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = bbox1_area + bbox2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def expand_bbox(bbox, margin):
    """Expand a bbox."""
    return [
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin
    ]

class InkMLFlowchartParser:
    def __init__(self, inkml_path, sam_model=None, sam_processor=None):
        self.tree = ET.parse(inkml_path)
        self.root = self.tree.getroot()
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        
        # namespace handling
        self.ns = {'inkml': 'http://www.w3.org/2003/InkML',
                   'fc': 'LUNAM/IRCCyN/FlowchartML'}
        
        self.traces = {}
        self.trace_groups = {}  # traceGroup info
        self.nodes = {}  # nodes parsed from FlowchartML
        self.arrows = {}  # arrows parsed from FlowchartML
        
        self.raw_min_x = float('inf')
        self.raw_max_x = float('-inf')
        self.raw_min_y = float('inf')
        self.raw_max_y = float('-inf')

        self._parse_traces()
        self._parse_trace_groups()
        self._parse_flowchart_gt()
        
        # compute scaling
        raw_w = self.raw_max_x - self.raw_min_x
        raw_h = self.raw_max_y - self.raw_min_y
        if raw_w <= 0: raw_w = 1
        if raw_h <= 0: raw_h = 1
        available_size = MAX_DIMENSION - 2 * PADDING
        self.scale_factor = available_size / max(raw_w, raw_h)
        self.img_w = int(raw_w * self.scale_factor + 2 * PADDING)
        self.img_h = int(raw_h * self.scale_factor + 2 * PADDING)

    def _parse_traces(self):
        """Parse all trace data."""
        for trace in self.root.findall('.//inkml:trace', self.ns):
            tid = trace.get('id')
            if not trace.text:
                continue
            points_str = trace.text.strip().split(',')
            points = []
            for p in points_str:
                coords = p.strip().split()
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
                    self.raw_min_x = min(self.raw_min_x, x)
                    self.raw_max_x = max(self.raw_max_x, x)
                    self.raw_min_y = min(self.raw_min_y, y)
                    self.raw_max_y = max(self.raw_max_y, y)
            self.traces[tid] = np.array(points, dtype=np.float32)

    def _parse_trace_groups(self):
        """Parse traceGroups, building an xml:id -> trace mapping."""
        root_group = self.root.find('.//inkml:traceGroup', self.ns)
        if root_group is None:
            return
        self._parse_trace_group_recursive(root_group)
    
    def _parse_trace_group_recursive(self, group):
        """Recursively parse traceGroups."""
        gid = group.get('{http://www.w3.org/XML/1998/namespace}id')
        if not gid:
            return
        
        truth_elem = group.find("./inkml:annotation[@type='truth']", self.ns)
        truth = truth_elem.text if truth_elem is not None else None
        
        trace_views = group.findall('./inkml:traceView', self.ns)
        trace_ids = [tv.get('traceDataRef') for tv in trace_views if tv.get('traceDataRef')]
        
        annotation_xml = group.find('./inkml:annotationXML', self.ns)
        href = None
        if annotation_xml is not None:
            href = annotation_xml.get('href')
            if href:
                href = href.replace('#', '') 
        
        raw_bbox = self.get_raw_bbox(trace_ids)
        if raw_bbox:
            cx = (raw_bbox[0] + raw_bbox[2]) / 2
            cy = (raw_bbox[1] + raw_bbox[3]) / 2
            raw_center = (cx, cy)
        else:
            raw_center = (0, 0)
        
        self.trace_groups[gid] = {
            'id': gid,
            'truth': truth,
            'trace_ids': trace_ids,
            'href': href,
            'raw_bbox': raw_bbox,
            'raw_center': raw_center
        }
        
        for child_group in group.findall('./inkml:traceGroup', self.ns):
            self._parse_trace_group_recursive(child_group)

    def _parse_flowchart_gt(self):
        """Parse the FlowchartML ground truth."""
        flowchart = self.root.find('.//fc:flowchart', self.ns)
        if flowchart is None:
            print("Warning: FlowchartML ground truth not found")
            return
        
        for node in flowchart.findall('./fc:node', self.ns):
            node_id = node.get('{http://www.w3.org/XML/1998/namespace}id')
            node_type = node.get('type')
            
            text_elem = node.find('./fc:text', self.ns)
            text_id = text_elem.get('{http://www.w3.org/XML/1998/namespace}id') if text_elem is not None else None
            text_content = text_elem.text if text_elem is not None else ""
            if text_content is None: text_content = ""
            
            self.nodes[node_id] = {
                'id': node_id,
                'type': node_type,
                'text_id': text_id,
                'text': text_content.strip()
            }
        
        for arrow in flowchart.findall('./fc:arrow', self.ns):
            arrow_id = arrow.get('{http://www.w3.org/XML/1998/namespace}id')
            source = arrow.get('source')
            target = arrow.get('target')
            
            text_elem = arrow.find('./fc:text', self.ns)
            text_id = text_elem.get('{http://www.w3.org/XML/1998/namespace}id') if text_elem is not None else None
            condition = text_elem.text if text_elem is not None else ""
            
            self.arrows[arrow_id] = {
                'id': arrow_id,
                'source': source,
                'target': target,
                'text_id': text_id,
                'condition': condition
            }

    def get_raw_bbox(self, trace_ids):
        all_pts = []
        for tid in trace_ids:
            if tid in self.traces:
                all_pts.append(self.traces[tid])
        if not all_pts:
            return None
        all_pts = np.vstack(all_pts)
        return [np.min(all_pts[:, 0]), np.min(all_pts[:, 1]),
                np.max(all_pts[:, 0]), np.max(all_pts[:, 1])]

    def to_pixel_coords(self, raw_x, raw_y):
        px = (raw_x - self.raw_min_x) * self.scale_factor + PADDING
        py = (raw_y - self.raw_min_y) * self.scale_factor + PADDING
        return int(px), int(py)

    def find_trace_group_by_href(self, href):
        for gid, group in self.trace_groups.items():
            if group['href'] == href:
                return group
        return None

    def has_valid_text_content(self):
        """Check whether all nodes have empty text."""
        if not self.nodes:
            return False
        
        for node in self.nodes.values():
            if node['text'] and len(node['text']) > 0:
                return True
        
        return False

    def get_arrowheads_from_sam3(self, image_path):
        """Detect arrowheads with SAM3."""
        if self.sam_model is None:
            return []
        try:
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
        except Exception as e:
            print(f"Error running SAM3: {e}")
            return []

    def generate_dataset(self, output_dir, output_json_name):
        """Build the dataset - detect arrowheads with SAM3."""
        os.makedirs(output_dir, exist_ok=True)
        img_save_dir = os.path.join(output_dir, "annotated_images")
        os.makedirs(img_save_dir, exist_ok=True)
        
        # 1. Render the base image
        base_img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        
        for tid, points in self.traces.items():
            scaled_points = []
            for pt in points:
                px, py = self.to_pixel_coords(pt[0], pt[1])
                scaled_points.append([px, py])
            pts = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(base_img, [pts], False, (0, 0, 0), LINE_THICKNESS, lineType=cv2.LINE_AA)
        
        # save a temp image for SAM3 detection
        temp_img_path = os.path.join(output_dir, "temp_base_image.jpg")
        cv2.imwrite(temp_img_path, base_img)
        
        # 2. Detect all arrowheads with SAM3
        detected_arrowheads = self.get_arrowheads_from_sam3(temp_img_path)
        
        if not detected_arrowheads:
            print(f"Warning: no arrowhead detected")
            # delete the temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return
        
        print(f"Detected {len(detected_arrowheads)} arrowheads")
        
        # 3. Prepare arrow data (from GT)
        # convert the arrow bbox to pixel coordinates
        arrow_pixel_bboxes = {}
        for arrow_id, arrow_data in self.arrows.items():
            arrow_group = self.find_trace_group_by_href(arrow_id)
            if not arrow_group or not arrow_group['raw_bbox']:
                continue
            
            raw_bbox = arrow_group['raw_bbox']
            bx1, by1 = self.to_pixel_coords(raw_bbox[0], raw_bbox[1])
            bx2, by2 = self.to_pixel_coords(raw_bbox[2], raw_bbox[3])
            
            arrow_pixel_bboxes[arrow_id] = {
                'bbox': [bx1, by1, bx2, by2],
                'data': arrow_data
            }
        
        # 4. For each arrowhead, find the best-matching arrow
        json_output = {"arrows": []}
        
        for arrowhead_idx, arrowhead in enumerate(detected_arrowheads):
            head_center = arrowhead['center']
            head_bbox = arrowhead['bbox']
            
            # find the arrow containing this arrowhead (its center inside the arrow bbox)
            best_arrow_id = None
            best_iou = 0.0
            
            for arrow_id, arrow_info in arrow_pixel_bboxes.items():
                arrow_bbox = arrow_info['bbox']
                
                # check whether the arrowhead center is inside the arrow bbox
                if (arrow_bbox[0] <= head_center[0] <= arrow_bbox[2] and 
                    arrow_bbox[1] <= head_center[1] <= arrow_bbox[3]):
                    
                    # use IoU as the match confidence
                    iou = bbox_iou(head_bbox, arrow_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_arrow_id = arrow_id
            
            # 5. Emit data (keep the arrowhead whether or not an arrow matched)
            if best_arrow_id:
                # matched an arrow; use the GT info
                arrow_data = arrow_pixel_bboxes[best_arrow_id]['data']
                source_id = arrow_data['source']
                target_id = arrow_data['target']
                condition = arrow_data['condition']
                
                source_text = self.nodes.get(source_id, {}).get('text', 'No Text')
                if not source_text:
                    source_text = 'No Text'
                target_text = self.nodes.get(target_id, {}).get('text', 'No Text')
                if not target_text:
                    target_text = 'No Text'
                
                arrow_id_str = f"matched_{best_arrow_id}"
            else:
                # no arrow matched; mark as unmatched
                source_text = "UNMATCHED"
                target_text = "UNMATCHED"
                condition = ""
                arrow_id_str = f"unmatched_{arrowhead_idx}"
                print(f"Warning: arrowhead {arrowhead_idx} matched no arrow")
            
            # 6. Draw the annotated image
            img_copy = base_img.copy()
            
            # draw the arrowhead bbox (blue, expanded 5px)
            expanded_head_bbox = expand_bbox(head_bbox, 5)
            bx1, by1 = int(expanded_head_bbox[0]), int(expanded_head_bbox[1])
            bx2, by2 = int(expanded_head_bbox[2]), int(expanded_head_bbox[3])
            
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(self.img_w, bx2), min(self.img_h, by2)
            
            cv2.rectangle(img_copy, (bx1, by1), (bx2, by2), (255, 0, 0), 3)
            
            # save the annotated image
            img_filename = f"arrow_{arrowhead_idx}_{arrow_id_str}.jpg"
            img_path = os.path.abspath(os.path.join(img_save_dir, img_filename))
            cv2.imwrite(img_path, img_copy)
            
            # 7. Save the JSON entry
            entry = {
                "arrow_id": f"arrow_{arrowhead_idx}",
                "start_point": [source_text],
                "end_point": [target_text],
                "condition": [condition] if condition else [],
                "matched_arrowhead_bbox": [float(head_bbox[0]), float(head_bbox[1]), 
                                          float(head_bbox[2]), float(head_bbox[3])],
                "annotated_image_path": img_path,
                "match_confidence": float(best_iou) if best_arrow_id else 0.0
            }
            json_output["arrows"].append(entry)
        
        # 8. Save the JSON
        json_path = os.path.join(output_dir, output_json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # delete the temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        matched_count = sum(1 for item in json_output['arrows'] if not item['arrow_id'].startswith('unmatched'))
        unmatched_count = len(json_output['arrows']) - matched_count
        
        print(f"Done! detected {len(json_output['arrows'])} arrowheads")
        print(f"  - matched: {matched_count}")
        print(f"  - unmatched: {unmatched_count}")
        print(f"JSON saved to: {json_path}")


def batch_process_inkml():
    """Batch-process InkML flowcharts."""
    inkml_dir = Path("<DATA_ROOT>/FC_A_raw")
    train_path = Path("<DATA_ROOT>/FC_A_raw/listInkML_Train.txt")
    output_dir = Path("<DATA_ROOT>/data_4_training/fca_2")
    sam_ckpt = "<MODEL_ROOT>/sam3/sam3.pt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # init the SAM3 model
    print("Loading SAM3 Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        sam_model = build_sam3_image_model(checkpoint_path=sam_ckpt, device=device)
        sam_processor = Sam3Processor(sam_model)
        print("SAM3 Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load SAM3: {e}")
        sam_model = None
        sam_processor = None
    
    try:
        with open(train_path, "r", encoding="utf-8") as f:
            file_keys = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: file not found {train_path}")
        return
    except Exception as e:
        print(f"Error reading file - {e}")
        return
    
    print(f"Found {len(file_keys)} files to process")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    removed_no_text_count = 0
    
    valid_file_keys = []
    
    for file_key in file_keys:
        base_name = file_key.replace('.inkml', '').replace('.jpeg', '').replace('.jpg', '')
        inkml_path = inkml_dir / f"{base_name}.inkml"
        
        if not inkml_path.exists():
            print(f"Skip: InkML file not found - {base_name}")
            skipped_count += 1
            continue
        
        try:
            file_output_dir = output_dir / base_name
            parser = InkMLFlowchartParser(str(inkml_path), sam_model, sam_processor)
            
            # check whether all nodes have empty text
            if not parser.has_valid_text_content():
                print(f"Removed: {base_name} - all nodes are 'No Text'; dropped and removed from the list.")
                removed_no_text_count += 1
                continue
            
            # emit data
            parser.generate_dataset(str(file_output_dir), f"{base_name}.json")
            
            valid_file_keys.append(file_key)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed: {processed_count} files")
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # rewrite the file list
    if removed_no_text_count > 0 or skipped_count > 0 or error_count > 0:
        print(f"\nUpdating the file list {train_path} ...")
        try:
            with open(train_path, "w", encoding="utf-8") as f:
                for key in valid_file_keys:
                    f.write(key + "\n")
            print("File list updated.")
        except Exception as e:
            print(f"Failed to update the file list: {e}")

    print("\n" + "="*50)
    print("Done!")
    print(f"Processed and kept: {processed_count} files")
    print(f"Removed for having no text: {removed_no_text_count} files")
    print(f"Missing/skipped: {skipped_count} files")
    print(f"Parse errors: {error_count} files")
    print(f"Original list length: {len(file_keys)}")
    print(f"New list length: {len(valid_file_keys)}")
    print(f"Output dir: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    batch_process_inkml()