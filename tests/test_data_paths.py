import tempfile
import unittest
from pathlib import Path
from gen_data.format_data import resolve_image_path

class ReleasedDataPathTests(unittest.TestCase):
    def test_relative_and_relocated_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'Dataset/data_4_training'
            sample = root / 'fcb/sample'
            image = sample / 'annotated_images/arrow.jpg'
            image.parent.mkdir(parents=True)
            image.touch()
            annotation = sample / 'sample.json'
            self.assertEqual(resolve_image_path('fcb/sample/annotated_images/arrow.jpg', annotation), str(image.resolve()))
            self.assertEqual(resolve_image_path('/old/machine/Dataset/data_4_training/fcb/sample/annotated_images/arrow.jpg', annotation), str(image.resolve()))
            self.assertEqual(resolve_image_path('missing.jpg', annotation), '')

if __name__ == '__main__':
    unittest.main()
