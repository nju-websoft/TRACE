import xml.etree.ElementTree as ET
import numpy as np
import json
import os
import cv2
import math
from pathlib import Path

# ================= Configuration =================
MAX_DIMENSION = 1333
PADDING = 20
LINE_THICKNESS = 2
# ===========================================

class InkMLParser:
    def __init__(self, inkml_path):
        self.tree = ET.parse(inkml_path)
        self.root = self.tree.getroot()
        for elem in self.root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        
        self.traces = {}
        self.entities = {}
        self.relations = []
        self.node_text_map = {}  # NodeID -> TextString mapping
        self.arrow_label_map = {}  # ArrowID -> Condition mapping
        
        self.raw_min_x = float('inf')
        self.raw_max_x = float('-inf')
        self.raw_min_y = float('inf')
        self.raw_max_y = float('-inf')

        self._parse_traces()
        self._parse_entities()
        self._parse_relations_and_text()
        
        # Compute scaling
        raw_w = self.raw_max_x - self.raw_min_x
        raw_h = self.raw_max_y - self.raw_min_y
        if raw_w <= 0: raw_w = 1
        if raw_h <= 0: raw_h = 1
        available_size = MAX_DIMENSION - 2 * PADDING
        self.scale_factor = available_size / max(raw_w, raw_h)
        self.img_w = int(raw_w * self.scale_factor + 2 * PADDING)
        self.img_h = int(raw_h * self.scale_factor + 2 * PADDING)
    def latex_to_keyboard(self,text):
        import re
        """
        Convert LaTeX math symbols to plain keyboard-typable characters
        """
        
        # Symbol map (simple direct replacements)
        symbol_map = {
            r'\cdot ': '·',
            r'\times': '×',
            r'\div': '÷',
            r'\pm': '±',
            r'\mp': '∓',
            r'\le': '<=',
            r'\leq': '<=',
            r'\ge': '>=',
            r'\geq': '>=',
            r'\ne': '!=',
            r'\neq': '!=',
            r'\lt': '<',
            r'\gt': '>',
            r'\approx': '≈',
            r'\equiv': '≡',
            r'\Delta': 'Delta',
            r'\delta': 'delta',
            r'\alpha': 'alpha',
            r'\beta': 'beta',
            r'\gamma': 'gamma',
            r'\theta': 'theta',
            r'\lambda': 'lambda',
            r'\mu': 'mu',
            r'\pi': 'pi',
            r'\sigma': 'sigma',
            r'\omega': 'omega',
            r'\infty': '∞',
            r'\sum': 'sum',
            r'\prod': 'prod',
            r'\int': 'integral',
            r'\lim': 'lim',
            r'\sin': 'sin',
            r'\cos': 'cos',
            r'\tan': 'tan',
            r'\log': 'log',
            r'\ln': 'ln',
            r'\exp': 'exp',
        }
        
        result = text
        if result is None:
            return ""
        
        # ===== Handle LaTeX commands that take arguments =====
        # Important: handle \sqrt before \frac so braces are not broken
        
        # Square root \sqrt{...} -> sqrt(...)
        def replace_sqrt(match):
            content = match.group(1)
            return f'sqrt({content})'
        
        result = re.sub(r'\\sqrt\{([^{}]+)\}', replace_sqrt, result)
        
        # Fraction \frac{a}{b} -> (a)/(b)
        # After sqrt is handled, numerator/denominator have no braces, so [^}]+ matches
        def replace_frac(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f'({numerator})/({denominator})'
        
        result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, result)
        
        # ===== Handle sub/superscripts =====
        
        # Superscript ^{...} or ^x
        result = re.sub(r'\^\{([^{}]+)\}', r'^\1', result)
        result = re.sub(r'\^(\w)', r'^\1', result)
        
        # Subscript _{...} or _x
        result = re.sub(r'_\{([^{}]+)\}', r'_\1', result)
        result = re.sub(r'_(\w)', r'_\1', result)
        
        # ===== Apply the symbol map =====
        for latex_sym, keyboard_sym in symbol_map.items():
            result = result.replace(latex_sym, keyboard_sym)
        
        # Remove dollar signs
        result = result.replace('$', '')
        
        # Collapse multiple backslashes into a space
        result = re.sub(r'\\\\+', ' ', result)
        
        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    def _parse_traces(self):
        for trace in self.root.findall('trace'):
            tid = trace.get('id')
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

    def _parse_entities(self):
        symbols = self.root.find('symbols')
        if symbols is None: return
        for group in symbols.findall('traceGroup'):
            gid = group.get('id')
            truth_annot = group.find("./annotation[@type='truth']")
            etype = truth_annot.text if truth_annot is not None else "unknown"
            
            meaning_annot = group.find("./annotation[@type='textMeaning']")
            text = meaning_annot.text if meaning_annot is not None else ""
            
            trace_views = group.findall("traceView")
            trace_ids = [tv.get('traceDataRef') for tv in trace_views]
            
            head_trace_ids = []
            head_elem = group.find("head")
            if head_elem:
                head_views = head_elem.findall("traceView")
                head_trace_ids = [tv.get('traceDataRef') for tv in head_views]
            
            # Parse the arrow connection points
            connection_from = None
            connection_to = None
            if etype == 'arrow':
                from_annot = group.find("./annotation[@type='connectionPointFrom']")
                to_annot = group.find("./annotation[@type='connectionPointTo']")
                if from_annot is not None and from_annot.text:
                    parts = from_annot.text.strip().split()
                    if len(parts) == 2:
                        trace_idx, point_idx = int(parts[0]), int(parts[1])
                        if trace_idx < len(trace_ids):
                            tid = trace_ids[trace_idx]
                            if tid in self.traces and point_idx < len(self.traces[tid]):
                                connection_from = tuple(self.traces[tid][point_idx])
                
                if to_annot is not None and to_annot.text:
                    parts = to_annot.text.strip().split()
                    if len(parts) == 2:
                        trace_idx, point_idx = int(parts[0]), int(parts[1])
                        if trace_idx < len(trace_ids):
                            tid = trace_ids[trace_idx]
                            if tid in self.traces and point_idx < len(self.traces[tid]):
                                connection_to = tuple(self.traces[tid][point_idx])
            
            raw_bbox = self.get_raw_bbox(trace_ids)
            if raw_bbox:
                cx = (raw_bbox[0] + raw_bbox[2]) / 2
                cy = (raw_bbox[1] + raw_bbox[3]) / 2
                raw_center = (cx, cy)
            else:
                raw_center = (0, 0)

            self.entities[gid] = {
                'id': gid, 'type': etype, 'text': text,
                'trace_ids': trace_ids, 'head_trace_ids': head_trace_ids,
                'raw_center': raw_center, 'raw_bbox': raw_bbox,
                'connection_from': connection_from,
                'connection_to': connection_to
            }

    def _parse_relations_and_text(self):
        """Parse topology relations, text-ownership relations, and arrow labels."""
        relations_node = self.root.find('relations')
        if relations_node is None: return

        for group in relations_node.findall('symbolGroup'):
            truth_elem = group.find("./annotation[@type='truth']")
            if truth_elem is None: continue
            
            rel_type = truth_elem.text
            views = group.findall('symbolView')
            ref_ids = [v.get('symbolDataRef') for v in views]
            
            # === Case 1: arrow_connection ===
            if rel_type == 'arrow_connection':
                arrow_id = None
                node_candidates = []
                for rid in ref_ids:
                    if rid in self.entities:
                        if self.entities[rid]['type'] == 'arrow':
                            arrow_id = rid
                        else:
                            node_candidates.append(rid)
                if arrow_id and len(node_candidates) == 2:
                    self.relations.append({'arrow_id': arrow_id, 'nodes': node_candidates})

            # === Case 2: text_inside / label ===
            elif rel_type in ['text_inside', 'label']:
                text_content = None
                container_id = None
                
                for rid in ref_ids:
                    if rid not in self.entities: continue
                    ent = self.entities[rid]
                    
                    if ent['type'] == 'text':
                        text_content = ent['text']
                    else:
                        container_id = rid
                
                if text_content and container_id:
                    self.node_text_map[container_id] = text_content
            
            # === Case 3: arrow_label ===
            elif rel_type == 'arrow_label':
                text_content = None
                arrow_id = None
                
                for rid in ref_ids:
                    if rid not in self.entities: continue
                    ent = self.entities[rid]
                    
                    if ent['type'] == 'text':
                        text_content = ent['text']
                    elif ent['type'] == 'arrow':
                        arrow_id = rid
                
                if text_content and arrow_id:
                    self.arrow_label_map[arrow_id] = text_content

    def get_node_text(self, node_id):
        """Get a node's text: prefer the text map, otherwise the entity's own text."""
        if node_id in self.node_text_map:
            return self.node_text_map[node_id]
        
        if node_id in self.entities:
            val = self.entities[node_id]['text']
            if val and val.strip(): return val
            
        return "No Text"

    def get_arrow_condition(self, arrow_id):
        """Get the arrow's condition label."""
        if arrow_id in self.arrow_label_map:
            return self.arrow_label_map[arrow_id]
        return ""

    def get_raw_bbox(self, trace_ids):
        all_pts = []
        for tid in trace_ids:
            if tid in self.traces: all_pts.append(self.traces[tid])
        if not all_pts: return None
        all_pts = np.vstack(all_pts)
        return [np.min(all_pts[:, 0]), np.min(all_pts[:, 1]),
                np.max(all_pts[:, 0]), np.max(all_pts[:, 1])]

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def to_pixel_coords(self, raw_x, raw_y):
        px = (raw_x - self.raw_min_x) * self.scale_factor + PADDING
        py = (raw_y - self.raw_min_y) * self.scale_factor + PADDING
        return int(px), int(py)

    def generate_dataset(self, output_dir, output_json_name):
        os.makedirs(output_dir, exist_ok=True)
        img_save_dir = os.path.join(output_dir, "annotated_images")
        os.makedirs(img_save_dir, exist_ok=True)
        
        base_img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        
        for tid, points in self.traces.items():
            scaled_points = []
            for pt in points:
                px, py = self.to_pixel_coords(pt[0], pt[1])
                scaled_points.append([px, py])
            pts = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(base_img, [pts], False, (0, 0, 0), LINE_THICKNESS, lineType=cv2.LINE_AA)

        # Save the clean whole-image (no annotations) for the E2E / triplet format
        stem = os.path.splitext(output_json_name)[0]
        whole_image_path = os.path.abspath(os.path.join(output_dir, f"{stem}.png"))
        cv2.imwrite(whole_image_path, base_img)

        json_output = {"image_path": whole_image_path, "arrows": []}
        
        for i, rel in enumerate(self.relations):
            arrow_data = self.entities[rel['arrow_id']]
            node_a = self.entities[rel['nodes'][0]]
            node_b = self.entities[rel['nodes'][1]]
            
            # Use connection points to decide direction (preferred)
            if arrow_data.get('connection_from') and arrow_data.get('connection_to'):
                # Use connectionPointFrom and connectionPointTo
                from_point = arrow_data['connection_from']
                to_point = arrow_data['connection_to']
                a = self.to_pixel_coords(from_point[0], from_point[1])
                b = self.to_pixel_coords(to_point[0], to_point[1])
                # Distance from each node center to from_point / to_point
                dist_a_from = self.get_distance(from_point, node_a['raw_center'])
                dist_b_from = self.get_distance(from_point, node_b['raw_center'])
                dist_a_to = self.get_distance(to_point, node_a['raw_center'])
                dist_b_to = self.get_distance(to_point, node_b['raw_center'])
                
                # The node nearer to from_point is the source
                # The node nearer to to_point is the target
                if dist_a_from < dist_b_from:
                    source_node = node_a
                    target_node = node_b
                else:
                    source_node = node_b
                    target_node = node_a
            else:
                # Fall back to the distance-based heuristic
                head_trace_ids = arrow_data['head_trace_ids'] or arrow_data['trace_ids']
                raw_head_bbox = self.get_raw_bbox(head_trace_ids)
                
                raw_head_center = ((raw_head_bbox[0]+raw_head_bbox[2])/2, (raw_head_bbox[1]+raw_head_bbox[3])/2)
                dist_a = self.get_distance(raw_head_center, node_a['raw_center'])
                dist_b = self.get_distance(raw_head_center, node_b['raw_center'])
                
                if dist_a < dist_b:
                    target_node, source_node = node_a, node_b
                else:
                    target_node, source_node = node_b, node_a
            
            # Get the arrowhead bbox (for annotation)
            head_trace_ids = arrow_data['head_trace_ids'] or arrow_data['trace_ids']
            raw_head_bbox = self.get_raw_bbox(head_trace_ids)
            
            # Get the cleaned text
            start_text = self.latex_to_keyboard(self.get_node_text(source_node['id']))
            end_text = self.latex_to_keyboard(self.get_node_text(target_node['id']))
            
            # Get the arrow condition
            condition = self.get_arrow_condition(arrow_data['id'])
                
            img_copy = base_img.copy()
            bx1, by1 = self.to_pixel_coords(raw_head_bbox[0], raw_head_bbox[1])
            bx2, by2 = self.to_pixel_coords(raw_head_bbox[2], raw_head_bbox[3])
            
            m = 5
            bx1, by1 = max(0, bx1-m), max(0, by1-m)
            bx2, by2 = min(self.img_w, bx2+m), min(self.img_h, by2+m)

            cv2.rectangle(img_copy, (bx1, by1), (bx2, by2), (255, 0, 0), 3)
            
            img_filename = f"arrow_{i}_{arrow_data['id']}.jpg"
            img_path = os.path.abspath(os.path.join(img_save_dir, img_filename))
            cv2.imwrite(img_path, img_copy)
            
            entry = {
                "arrow_id": int(arrow_data['id']) if arrow_data['id'].isdigit() else arrow_data['id'],
                "source_node": [start_text],
                "target_node": [end_text],
                "condition": [condition] if condition else [],
                "matched_arrowhead_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                "annotated_image_path": img_path
            }
            json_output["arrows"].append(entry)
            
        with open(os.path.join(output_dir, output_json_name), 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
            
        print(f"Done. Processed {len(json_output['arrows'])} arrows.")


def batch_process_inkml():
    """Batch-process InkML flowcharts."""
    # Paths
    inkml_dir = Path("<DATA_ROOT>/FC_B_raw")
    train_path = Path("<DATA_ROOT>/FC_B_raw/FC_Test.txt")
    output_dir = Path("<DATA_ROOT>/data_4_training/fcb_test")
    
    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the split list
    try:
        with open(train_path, "r", encoding="utf-8") as f:
            # Read all lines, dropping empties and surrounding whitespace
            file_keys = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: file not found {train_path}")
        return
    except Exception as e:
        print(f"Error reading file - {e}")
        return
    
    print(f"Found {len(file_keys)} files to process")
    
    # Stats
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each file
    for file_key in file_keys:
        # Strip any extension
        base_name = file_key.replace('.inkml', '').replace('.jpeg', '').replace('.jpg', '')
        
        inkml_path = inkml_dir / f"{base_name}.inkml"
        
        # Check the InkML file exists
        if not inkml_path.exists():
            print(f"Skip: InkML file missing - {base_name}")
            skipped_count += 1
            continue
        
        try:
            # Per-file output directory
            file_output_dir = output_dir / base_name
            
            # Parse and generate data
            parser = InkMLParser(str(inkml_path))
            parser.generate_dataset(str(file_output_dir), f"{base_name}.json")
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed: {processed_count} files")
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            error_count += 1
    
    # Print stats
    print("\n" + "="*50)
    print("Done!")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output dir: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    batch_process_inkml()