"""Freeze a Llama-2-only controlled diagnostic; never read final test questions."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parents[2]
sys.path[:0] = [str(PHASE), str(PHASE.parent), str(ROOT)]
from common import views
from audit_raw_data import group_id
from src.mcq_symmetrization import mcq_choice_texts

REF = ROOT / 'experiments/caa_reproduction/reference'
SCRATCH = Path('/scratch/rai/vast1/u1557683/experiments/phase_b_caa_behavior_20261022')
RESULT = ROOT / 'results/corrigibility_rca_20261027'
OUT = SCRATCH / 'corrigibility_rca_20261027'

def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + '\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    train_path = PHASE / 'data/corrigible_more_train.json'
    val_path = PHASE / 'data/corrigible_more_validation.json'
    train, val = [json.loads(p.read_text()) for p in (train_path, val_path)]
    original_path = REF / 'datasets/test/corrigible-neutral-HHH/test_dataset_ab.json'
    oldtrain_path = REF / 'datasets/generate/corrigible-neutral-HHH/generate_dataset.json'
    original, oldtrain = [json.loads(p.read_text()) for p in (original_path, oldtrain_path)]
    canonical = lambda v: group_id(v['question'], True)
    old_ids = {canonical(v) for v in original}
    val_ids = {g['group_id'] for g in val}
    train_ids = {g['group_id'] for g in train}
    assert None not in old_ids and not old_ids & train_ids
    assert not val_ids & train_ids
    assert not old_ids & {canonical(v) for v in oldtrain}
    assert not val_ids & {canonical(v) for v in oldtrain}
    examples = []
    for i,v in enumerate(original):
        g = {'views':[v]}
        for vi,w in enumerate(views(g)):
            examples.append(dict(dataset='original50' if vi==0 else 'original50_swapped',
                group_id=canonical(v), group_index=i, variant=vi, **w))
    for i,g in enumerate(val):
        for vi,v in enumerate(views(g)):
            examples.append(dict(dataset='new_validation206', group_id=g['group_id'], group_index=i, variant=vi, **v))
    audit_rows = []
    for split, groups in [('train',train), ('validation',val)]:
        for i,g in enumerate(groups):
            positive_texts = set()
            for v in g['views']:
                _, a, b = mcq_choice_texts(v['question'])
                assert {v['answer_matching_behavior'],v['answer_not_matching_behavior']} == {'(A)','(B)'}
                positive_texts.add(a if v['answer_matching_behavior']=='(A)' else b)
            assert len(positive_texts)==1
            sv = views(g)
            _, a0, b0 = mcq_choice_texts(sv[0]['question'])
            _, a1, b1 = mcq_choice_texts(sv[1]['question'])
            assert (a0,b0)==(b1,a1)
            assert sv[0]['answer_matching_behavior']!=sv[1]['answer_matching_behavior']
            audit_rows.append(dict(split=split,index=i,group_id=g['group_id'],
                first_view_target=g['views'][0]['answer_matching_behavior'],
                positive_text=next(iter(positive_texts)),swap_text_and_label_passed=True))
    with (RESULT/'label_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit_rows[0]));w.writeheader();w.writerows(audit_rows)
    save(HERE/'evaluation_manifest.json',examples)
    save(HERE/'training_manifest.json',train[:256])
    counts={str(n):dict(Counter(g['views'][0]['answer_matching_behavior'] for g in train[:n])) for n in [12,64,256]}
    checks={'canonical_train_validation_overlap':len(train_ids & val_ids),
        'new_train_original50_overlap':len(train_ids & old_ids),
        'original_train_new_validation_overlap':len(val_ids & {canonical(v) for v in oldtrain}),
        'final_test_questions_opened':False,'source_mapping':'Frozen prepare_data.py maps raw label +1 to matching and -1 to non-matching; no remapping here.',
        'structural_label_checks_passed':True,'semantic_labels_exhaustively_validated':False,
        'first_orientation_target_counts':counts,
        'validation_groups':len(val),'original_questions':len(original),
        'semantic_overlap_limitation':'Canonical exact-text/whitespace/swap checks do not rule out paraphrases.',
        'inputs':{str(p):digest(p) for p in [train_path,val_path,original_path,oldtrain_path,PHASE/'config.json',ROOT/'experiments/caa_reproduction/linear_eval.py']}}
    save(RESULT/'data_audit.json',checks)
    save(HERE/'config.json',{'scope':'Llama-2 Corrigibility root-cause diagnostic only',
        'model':'meta-llama/Llama-2-7b-chat-hf','revision':'f5db02db724555f92da89c216ac04704f23d4590',
        'zero_based_decoder_block':13,'hidden_state_index':14,'alphas':[-1,0,1],
        'dtype':'float32','tf32':False,'attention':'eager','no_hook_baseline':True,
        'primary_2x2':'original normalized / new pilot12 vector x original50 / original new pilot2 groups (4 views)',
        'supplementary_evaluation':'all 206 frozen validation groups, both orientations; original50 text swaps',
        'training_sizes':[12,64,256],'training_selection':'first n frozen train groups, no outcome selection',
        'orientation_ablation':'12 and 256 groups with both literal source orientations; same fixed original CAA norm',
        'norm_ablation':'cross original/new12 directions with original seven-behavior norm and pilot four-concept norm',
        'larger_train_norm':'original CAA norm held fixed; no other concepts/models rerun',
        'scorers':['full-vocabulary target probability','A/B-normalized target probability'],
        'pipeline_bridge':'official CAA wrapper and new prompt/hook on identical weights under transformers4.33.2, then actual pilot runtime5.6.2',
        'test_used':False,'full_sweep':False,'github_push':False,'output_root':str(OUT),
        'frozen_evaluation_sha256':digest(HERE/'evaluation_manifest.json'),
        'frozen_training_sha256':digest(HERE/'training_manifest.json')})
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
