"""Frozen inputs and adapters for the positive-only small-lambda study."""
import hashlib, importlib.util, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/small_lambda_positive_20261110'
spec = importlib.util.spec_from_file_location('original_style_reference', ROOT/'experiments/qwen_original_style_20261105/run.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
stage = original.prior.stage
SPLIT = original.prior.SPLIT
from runtime_b import task, data
from weighted_small import AttenuatedCurveball
WEIGHTS = [0., .001, .005, .01, .05, .1, .25, .5, 1.]
NAMES = {'coordinate-other-ais':'AI Coordination','corrigible-neutral-HHH':'Corrigibility','hallucination':'Hallucination','myopic-reward':'Myopic Reward','survival-instinct':'Survival Instinct','sycophancy':'Sycophancy','refusal':'Refusal'}
load = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def config():return load(OUT/'experiment_config.json')
def fit_path(key,c,w):return OUT/'fits'/key/f'{c}_lambda_{w}.joblib'
def verify():
    cfg=config()
    for p,h in cfg['frozen_inputs'].items():assert sha(ROOT/p)==h,p
    return cfg
