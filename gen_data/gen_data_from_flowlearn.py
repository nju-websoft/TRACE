# Generate arrow triplets from SVG
# FlowLearn uses a different mermaid/SVG version than FlowVQA, so the SVG-parsing logic differs
import xml.etree.ElementTree as ET
import re
import math
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

def parse_svg_flowchart_corrected(svg_content):
    """Parse an SVG flowchart and extract node and arrow information."""
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'xhtml': 'http://www.w3.org/1999/xhtml' 
    }
    
    root = ET.fromstring(svg_content)
    
    # Get the ViewBox offset
    viewbox = root.get('viewBox')
    offset_x, offset_y = 0.0, 0.0
    if viewbox:
        vb_parts = [float(x) for x in re.split(r'[,\s]+', viewbox.strip())]
        if len(vb_parts) >= 2:
            offset_x = vb_parts[0]
            offset_y = vb_parts[1]

    def to_img_coords(x, y):
        return x - offset_x, y - offset_y

    # Extract nodes
    nodes_map = {}
    
    for node in root.findall(".//svg:g", ns):
        class_attr = node.get('class', '')
        if 'node' in class_attr and 'default' in class_attr:
            full_id = node.get('id', '')
            match = re.search(r'flowchart-(entity\d+)-\d+', full_id)
            if not match:
                continue
            entity_key = match.group(1)
            
            # Extract text
            node_text = ""
            text_element = node.find(".//xhtml:span[@class='nodeLabel']", ns)
            
            if text_element is not None:
                node_text = "".join(text_element.itertext())
            else:
                svg_text_element = node.find(".//svg:text", ns)
                if svg_text_element is not None:
                    node_text = "".join(svg_text_element.itertext())

            # Extract coordinates
            transform = node.get('transform', '')
            trans_match = re.search(r'translate\(([^,]+),\s*([^\)]+)\)', transform)
            tx, ty = (float(trans_match.group(1)), float(trans_match.group(2))) if trans_match else (0.0, 0.0)
            
            # Extract rectangle size
            rect = node.find(".//svg:rect", ns)
            if rect is not None:
                rx = float(rect.get('x', 0))
                ry = float(rect.get('y', 0))
                rw = float(rect.get('width', 0))
                rh = float(rect.get('height', 0))
                
                abs_x1 = tx + rx
                abs_y1 = ty + ry
                abs_x2 = abs_x1 + rw
                abs_y2 = abs_y1 + rh
                
                img_x1, img_y1 = to_img_coords(abs_x1, abs_y1)
                img_x2, img_y2 = to_img_coords(abs_x2, abs_y2)
                
                nodes_map[entity_key] = {
                    "text": node_text,
                    "bbox": [round(img_x1, 1), round(img_y1, 1), round(img_x2, 1), round(img_y2, 1)]
                }

    # Extract links and arrows
    arrows_data = []
    arrow_id_counter = 1
    
    for path in root.findall(".//svg:path", ns):
        class_attr = path.get('class', '')
        if 'flowchart-link' not in class_attr:
            continue
            
        start_match = re.search(r'LS-(entity\d+)', class_attr)
        end_match = re.search(r'LE-(entity\d+)', class_attr)
        
        start_text = [nodes_map[start_match.group(1)]['text']] if (start_match and start_match.group(1) in nodes_map) else ["Unknown"]
        end_text = [nodes_map[end_match.group(1)]['text']] if (end_match and end_match.group(1) in nodes_map) else ["Unknown"]
            
        d_str = path.get('d', '')
        arrow_bbox = calculate_arrowhead_bbox(d_str, offset_x, offset_y)
        
        arrows_data.append({
            "arrow_id": arrow_id_counter,
            "source_node": start_text,
            "target_node": end_text,
            "condition": [],
            "matched_arrowhead_bbox": arrow_bbox
        })
        arrow_id_counter += 1
        
    return arrows_data


def calculate_arrowhead_bbox(d_string, off_x, off_y):
    """Compute the arrowhead bounding box."""
    points = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', d_string)
    points = [float(p) for p in points]
    
    if len(points) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    
    tip_x = points[-2]
    tip_y = points[-1]
    prev_x = points[-4]
    prev_y = points[-3]
    
    dx = tip_x - prev_x
    dy = tip_y - prev_y
    angle = math.atan2(dy, dx)
    
    arrow_len = 10.0
    arrow_half_width = 5.0
    
    local_points = [
        (0, 0),
        (-arrow_len, -arrow_half_width),
        (-arrow_len, arrow_half_width)
    ]
    
    final_xs = []
    final_ys = []
    
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    for lx, ly in local_points:
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        
        final_x = (rx + tip_x) - off_x
        final_y = (ry + tip_y) - off_y
        
        final_xs.append(final_x)
        final_ys.append(final_y)
        
    return [
        round(min(final_xs), 1), 
        round(min(final_ys), 1), 
        round(max(final_xs), 1), 
        round(max(final_ys), 1)
    ]


def draw_arrow_annotations(image_path, arrows_data, output_dir, base_name):
    """
    Draw an annotated image for each arrow and return the image paths.

    Args:
        image_path: path to the source image
        arrows_data: list of arrow records
        output_dir: output directory
        base_name: base file name

    Returns:
        the arrow records updated with image paths
    """
    try:
        # Open the source image
        img = Image.open(image_path).convert('RGB')
        
        img_width, img_height = img.size

        for arrow in arrows_data:
            # Copy the source image
            annotated_img = img.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Get the arrowhead bbox
            bbox = arrow['matched_arrowhead_bbox']
            x1, y1, x2, y2 = bbox
            
            # Expand the box (5px padding)
            x1_pad = x1 - 5
            y1_pad = y1 - 5
            x2_pad = x2 + 5
            y2_pad = y2 + 5
            
            # Clamp coordinates to [0, width/height]
            x1_clamped = max(0, min(x1_pad, img_width))
            y1_clamped = max(0, min(y1_pad, img_height))
            x2_clamped = max(0, min(x2_pad, img_width))
            y2_clamped = max(0, min(y2_pad, img_height))
            
            # Ensure x1 <= x2 and y1 <= y2 (avoid inverted rectangle)
            final_x1 = min(x1_clamped, x2_clamped)
            final_y1 = min(y1_clamped, y2_clamped)
            final_x2 = max(x1_clamped, x2_clamped)
            final_y2 = max(y1_clamped, y2_clamped)
            
            # Draw a blue rectangle, width 3
            draw.rectangle(
                [(final_x1, final_y1), (final_x2, final_y2)],
                outline='blue',
                width=3
            )
            
            # Save the annotated image
            arrow_id = arrow['arrow_id']
            output_path = output_dir / f"arrow_{arrow_id}.jpg"
            annotated_img.save(output_path, 'JPEG', quality=95)
            
            # Store the absolute image path in the arrow record
            arrow['annotated_image_path'] = str(output_path.absolute())

        return arrows_data
            
    except Exception as e:
        raise Exception(f"Error drawing arrow annotations: {e}")


def batch_process_flowcharts():
    """Batch-process flowcharts."""
    # Paths
    jpeg_dir = Path("<DATA_ROOT>/flowlearn/mermaid_word/jpeg")
    svg_dir = Path("<DATA_ROOT>/flowlearn/mermaid_word/svg")
    train_json_path = Path("<DATA_ROOT>/flowlearn/test.json")
    output_dir = Path("<DATA_ROOT>/data_4_training/flowlearn_test")
    
    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the split json
    try:
        with open(train_json_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found {train_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: {train_json_path} is not valid JSON")
        return
    
    # Get the list of files to process
    if isinstance(train_data, dict):
        file_keys = list(train_data.keys())
    elif isinstance(train_data, list):
        file_keys = train_data
    else:
        print("Error: unsupported json format")
        return
    
    print(f"Found {len(file_keys)} files to process")
    
    # Stats
    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_arrows = 0
    
    # Process each file
    for file_key in file_keys:
        # Strip the .jpeg suffix if present
        base_name = file_key.replace('.jpeg', '').replace('.jpg', '')
        
        jpeg_path = jpeg_dir / f"{base_name}.jpeg"
        svg_path = svg_dir / f"{base_name}.svg"
        
        # Per-file output folder
        file_output_dir = output_dir / base_name
        file_output_dir.mkdir(exist_ok=True)
        
        annotations_dir = file_output_dir / "annotated_images"
        annotations_dir.mkdir(exist_ok=True)
        
        output_path = file_output_dir / f"{base_name}.json"
        
        # Check the JPEG exists
        if not jpeg_path.exists():
            # Try the .jpg suffix
            jpeg_path = jpeg_dir / f"{base_name}.jpg"
            if not jpeg_path.exists():
                print(f"Skip: JPEG missing - {base_name}")
                skipped_count += 1
                continue
        
        # Check the SVG exists
        if not svg_path.exists():
            print(f"Skip: SVG missing - {base_name}")
            skipped_count += 1
            continue
        
        try:
            # Read and parse the SVG
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
            
            arrows = parse_svg_flowchart_corrected(svg_content)
            
            # Draw annotated images and update arrow records with paths
            arrows = draw_arrow_annotations(jpeg_path, arrows, annotations_dir, base_name)
            
            # Build the output data
            output_data = {
                "arrows": arrows
            }
            
            # Save the JSON result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            total_arrows += len(arrows)
            
            if processed_count % 10 == 0:
                print(f"Processed: {processed_count} files, {total_arrows} arrows")
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            error_count += 1
    
    # Print stats
    print("\n" + "="*50)
    print("Done!")
    print(f"Processed: {processed_count} files")
    print(f"Total arrows: {total_arrows}")
    print(f"Skipped: {skipped_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output dir: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    batch_process_flowcharts()