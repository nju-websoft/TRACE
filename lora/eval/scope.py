"""Select the intended evaluation split independently of prediction completeness."""
import json
from pathlib import Path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

def select_ground_truth(ground_truth, image_dir=None, test_json=None):
    expected = None
    if test_json:
        data = json.loads(Path(test_json).read_text())
        records = data.items() if isinstance(data, dict) else enumerate(data)
        expected = set()
        for key, record in records:
            if isinstance(record, dict):
                identifier = next((record.get(field) for field in ('id', 'image_name', 'file_name', 'image', 'image_path') if record.get(field) is not None), key)
            elif isinstance(record, str):
                identifier = record
            else:
                identifier = key
            expected.add(Path(str(identifier)).stem)
    elif image_dir:
        directory = Path(image_dir)
        if not directory.is_dir():
            raise ValueError(f'Evaluation image directory does not exist: {directory}')
        expected = {p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    if expected is None:
        if not ground_truth:
            raise ValueError('No ground truth was loaded. Check the dataset paths.')
        return ground_truth
    if not expected:
        raise ValueError('The requested evaluation split is empty.')
    selected = {key: value for key, value in ground_truth.items() if Path(str(key)).stem in expected}
    missing = expected - {Path(str(key)).stem for key in selected}
    if missing:
        raise ValueError(f'Ground truth is missing for {len(missing)} requested images; check split paths and image IDs.')
    return selected
