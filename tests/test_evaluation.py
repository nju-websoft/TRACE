import unittest
import tempfile
import json
from pathlib import Path
from lora.eval import eval_TRACE, eval_E2E
from lora.eval.scope import select_ground_truth

class EvaluationRegressionTests(unittest.TestCase):
    def test_split_scope_does_not_include_training_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test_one.png').touch()
            (root / 'test_two.jpg').touch()
            gt = {'train_one': set(), 'test_one': set(), 'test_two': set()}
            self.assertEqual(set(select_ground_truth(gt, root)), {'test_one', 'test_two'})
            split = root / 'test.json'
            split.write_text(json.dumps([{'id':'test_one'}]))
            self.assertEqual(set(select_ground_truth(gt, root, split)), {'test_one'})

    def test_missing_image_is_counted_as_false_negative(self):
        triple = ('start', 'connectedto', 'end')
        for evaluator in (eval_TRACE, eval_E2E):
            result = evaluator._evaluate_simple('example', {'one': {triple}}, {'one': {triple}, 'two': {triple}})
            self.assertEqual(result['total_gt'], 2)
            self.assertEqual(result['missing_prediction_files'], 1)
            self.assertAlmostEqual(result['standard_metrics']['recall'], .5)
            self.assertAlmostEqual(result['standard_metrics']['f1'], 2 / 3)

    def test_similarity_boundary_matches_paper(self):
        for evaluator in (eval_TRACE, eval_E2E):
            pred = {('abcdefghijklmnopqrst', 'yes', 'end')}
            gt = {('abcdefghijklmnopqXYZ', 'yes', 'end')}
            self.assertAlmostEqual(evaluator.edit_similarity(next(iter(pred))[0], next(iter(gt))[0]), .85)
            self.assertEqual(evaluator._relaxed_match(pred, gt)[:2], (0, 1))

    def test_containment_missing_image(self):
        triple = ('task', 'partof', 'lane')
        for evaluator in (eval_TRACE, eval_E2E):
            result = evaluator._evaluate_with_containment('example', {}, {'one': {triple}})
            self.assertEqual(result['total_gt'], 1)
            self.assertEqual(result['missing_prediction_files'], 1)
            self.assertEqual(result['overall']['standard']['recall'], 0)

if __name__ == '__main__':
    unittest.main()
