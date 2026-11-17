"""Read-only access to the completed small-lambda experiment."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/small_lambda_validation_test_20261110'
sys.path.insert(0,str(ROOT/'experiments/small_lambda_positive_20261110'))
import study_common as prev
import execute as old_run
OLD=prev.OUT
NAMES=prev.NAMES
load=prev.load;save=prev.save;sha=prev.sha
def verify():
    cfg=load(OUT/'frozen_config.json')
    for p,h in cfg['protected_inputs'].items():assert sha(ROOT/p)==h,p
    return cfg
