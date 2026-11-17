"""Safety checks for automatic resubmission; no scheduler or filesystem writes."""
import unittest
from unittest.mock import patch
import recovery_controller as r

class RecoveryTests(unittest.TestCase):
    def test_only_infrastructure_retries(self):
        for state in ['PREEMPTED','NODE_FAIL','BOOT_FAIL','TIMEOUT']:
            self.assertEqual(r.decision(state),'retry')
        for state in ['FAILED','OUT_OF_MEMORY','CANCELLED']:
            self.assertEqual(r.decision(state),'stop')
    def test_science_failure_never_retried(self):
        self.assertEqual(r.decision('PREEMPTED',True),'stop')
    def test_wait_for_active_and_unknown(self):
        for state in ['PENDING','RUNNING','COMPLETING','UNKNOWN']:
            self.assertEqual(r.decision(state),'wait')
    def test_completed_requires_artifacts(self):
        self.assertEqual(r.decision('COMPLETED'),'verify')
    def test_no_duplicate_submission(self):
        with patch.object(r,'active_jobs',return_value=[('123','RUNNING')]),patch.object(r,'command') as cmd:
            r.submit_gpu({'gpu_type':'a40'},'notch474');cmd.assert_not_called()
    def test_uncertain_submission_stops(self):
        with patch.object(r,'active_jobs',return_value=[]),patch.object(r,'command') as cmd:
            with self.assertRaises(RuntimeError):r.submit_gpu({'submission_uncertain':True},'notch474')
            cmd.assert_not_called()
    def test_rotates_compatible_gpu_and_preserves_script(self):
        state={'gpu_type':'a40','retries':0}
        with patch.object(r,'active_jobs',return_value=[]),patch.object(r,'save'),patch.object(r,'event'),patch.object(r,'command',return_value='456') as cmd:
            r.submit_gpu(state,'notch474')
            args=cmd.call_args.args[0]
            self.assertIn('--gres=gpu:a6000:1',args)
            self.assertIn('--exclude=notch474',args)
            self.assertEqual(args[-1],str(r.GPU_SCRIPT))
            self.assertEqual(state['job_id'],'456')
            self.assertFalse(state['submission_uncertain'])

if __name__=='__main__':unittest.main()
