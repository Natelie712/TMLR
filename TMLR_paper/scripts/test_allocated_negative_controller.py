import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import allocated_negative_controller as a

class Tests(unittest.TestCase):
    def test_allocation_not_step_status(self):
        with patch.object(a,'command',return_value='123|RUNNING|\n123.0|FAILED|'):
            self.assertEqual(a.allocation_state(dict(cluster='granite',allocation='123')),'RUNNING')
    def test_only_own_steps(self):
        with patch.object(a,'command',return_value='123.0|interactive\n123.1|tmlr-negative-search-step\n'):
            self.assertEqual(a.steps(dict(cluster='granite',allocation='123')),['123.1'])
    def test_no_duplicate_fallback(self):
        with patch.object(a,'command',return_value='456') as cmd:
            with self.assertRaises(RuntimeError):a.fallback({},'NODE_FAIL')
            self.assertEqual(cmd.call_count,1)
    def test_ambiguous_submission_never_repeated(self):
        with patch.object(a,'command',return_value='') as cmd:
            with self.assertRaises(RuntimeError):a.fallback(dict(fallback_submission_uncertain=True),'TIMEOUT')
            self.assertEqual(cmd.call_count,1)

if __name__=='__main__':unittest.main()
