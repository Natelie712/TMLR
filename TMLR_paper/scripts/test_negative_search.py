"""Scientific selection and resume tests; no model inference."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import negative_search as n

class Tests(unittest.TestCase):
    def test_select_suppression_not_increase(self):
        scores={w:2. for w in n.WEIGHTS};scores[.01]=-7.;scores[.5]=9.
        self.assertEqual(n.choose(scores),.01)
    def test_exact_tie_prefers_smaller_lambda(self):
        self.assertEqual(n.choose({w:-2. for w in n.WEIGHTS}),0.)
    def test_all_wrong_direction_kept(self):
        scores={w:5.+w for w in n.WEIGHTS};self.assertEqual(n.choose(scores),0.)
    def test_incomplete_grid_rejected(self):
        with self.assertRaises(AssertionError):n.choose({0.:-4.})
    def test_nonfinite_rejected(self):
        scores={w:0. for w in n.WEIGHTS};scores[1.]=float('nan')
        with self.assertRaises(AssertionError):n.choose(scores)
    def test_checkpoint_truncates_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'config.json';plan.write_text('{}');p=root/'rows/qwen3/c_validation.jsonl';p.parent.mkdir(parents=True)
            obj=dict(config_sha256=n.sha(plan),row=dict(index=1,method='Linear'))
            valid=json.dumps(obj)+'\n';p.write_text(valid+'{"incomplete":')
            with patch.object(n,'DEST',root),patch.object(n,'PLAN',plan):
                _,rows=n.checkpoint('qwen3','c','validation')
            self.assertEqual(p.read_text(),valid);self.assertEqual(len(rows),1)
    def test_incompatible_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'config.json';plan.write_text('{}');p=root/'rows/qwen3/c_validation.jsonl';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(config_sha256='wrong',row=dict(index=1,method='Linear')))+'\n')
            with patch.object(n,'DEST',root),patch.object(n,'PLAN',plan):
                with self.assertRaises(AssertionError):n.checkpoint('qwen3','c','validation')

if __name__=='__main__':unittest.main()
