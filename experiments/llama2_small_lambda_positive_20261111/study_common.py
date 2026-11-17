"""Frozen inputs and adapters for the positive-only small-lambda study."""
import hashlib, importlib.util, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/llama2_small_lambda_positive_20261111'
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

def read_rows(concept,partition):
    p=OUT/'rows'/'llama2'/f'{concept}_{partition}.jsonl'
    rows=[]
    if p.exists():
        for line in p.read_text().splitlines():
            try:obj=json.loads(line)
            except json.JSONDecodeError:break
            assert obj['config_sha256']==sha(OUT/'experiment_config.json')
            rows.extend(obj['rows'])
    return rows

def methods_for(c,partition):
    if partition!='benchmark':weights=WEIGHTS
    else:
        p=OUT/'selected_configurations.json'
        weights=sorted({0.,1.,load(p)['selected'][c]}) if p.exists() else [0.,1.]
    return ['No-hook','Linear']+[f'lambda_{w}' for w in weights]

def freeze_selection():
    import numpy as np
    path=OUT/'selected_configurations.json'
    if path.exists():return verify_selection()
    assert not any((OUT/'rows'/'llama2').glob('*_benchmark.jsonl')),'Test inference must follow selection freeze'
    selected={};details=[];inputs={}
    for c in NAMES:
        rows=read_rows(c,'validation');lookup={(r['index'],r['method']):r for r in rows}
        assert len(lookup)==len(rows)
        ids=sorted(r['index'] for r in SPLIT if r['concept']==c and r['partition']=='validation')
        assert set(lookup)=={(i,m) for i in ids for m in methods_for(c,'validation')}
        base=np.array([lookup[i,'No-hook']['normalized_target_probability'] for i in ids])
        values={w:float(np.mean([lookup[i,f'lambda_{w}']['normalized_target_probability'] for i in ids])-base.mean())*100 for w in config()['selection_eligible_weights']}
        best=max(values.values());chosen=min(w for w,v in values.items() if best-v<=1e-6)
        selected[c]=chosen;details.append(dict(concept=c,selected_weight=chosen,validation_delta_pp=values[chosen],eligible_validation_means=values))
        p=OUT/'rows'/'llama2'/f'{c}_validation.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in (OUT/'fits'/'llama2').glob('*'):
        if p.is_file():inputs[str(p.relative_to(ROOT))]=sha(p)
    save(path,dict(selected=selected,details=details,rule=config()['selection_rule'],config_sha256=sha(OUT/'experiment_config.json'),protected_inputs=inputs,exploratory=True,test_used_for_selection=False))
    return verify_selection()

def verify_selection():
    frozen=load(OUT/'selected_configurations.json')
    assert frozen['config_sha256']==sha(OUT/'experiment_config.json')
    assert set(frozen['selected'])==set(NAMES)
    for p,h in frozen['protected_inputs'].items():assert sha(ROOT/p)==h,p
    return frozen
