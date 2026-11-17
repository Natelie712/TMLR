"""Freeze the full-data Linear-only transfer protocol before model evaluation."""
import hashlib,json
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PHASE=ROOT/'experiments/curveball_table1_four_models_20261022/phase_b_caa_behavior'
RESULT=ROOT/'results/caa_transfer_full_20261027'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n')

def main():
    RESULT.mkdir(exist_ok=True)
    old=json.loads((PHASE/'config.json').read_text());meta=json.loads((PHASE/'data/manifest.json').read_text())
    seen=defaultdict(set);counts={};hashes={}
    for c in old['concepts']:
        counts[c]={}
        for split in ['train','validation','test']:
            p=PHASE/'data'/f'{c}_{split}.json';groups=json.loads(p.read_text())
            assert sha(p)==meta['datasets'][c]['splits'][split]['sha256']
            assert len({g['group_id'] for g in groups})==len(groups)
            for g in groups:
                seen[g['group_id']].add(split)
                for v in g['views']:assert {v['answer_matching_behavior'],v['answer_not_matching_behavior']}=={'(A)','(B)'}
            counts[c][split]=dict(groups=len(groups),source_orientations=sum(len(g['views']) for g in groups))
            hashes[str(p)]=sha(p)
    assert all(len(s)==1 for s in seen.values()),'Cross-concept split leakage'
    cfg=dict(protocol_id='caa_linear_full_transfer_v1_20261027',models=old['models'],concepts=old['concepts'],
        split_source=str(PHASE/'data'),split_counts=counts,split_sha256=hashes,seed=42,
        normalization=dict(id='CAA_per_layer_mean_raw_norm_own_four_concepts_full_train',
            formula='v[c,l] = raw[c,l] * mean_c(norm(raw[c,l])) / norm(raw[c,l])',
            reference_pool=old['concepts'],reference_pool_size=4,
            adaptation='Original CAA uses seven original behaviors; only four own-data behavioral concepts exist in this stage. Preserve the per-layer cross-behavior mean-norm formula, explicitly adapt the pool. Never substitute the old fixed Llama-2 norm or unit norm.',
            alternative_calibrations_run=False),
        training=dict(groups='all frozen train groups',orientations='all distinct valid source orientations',
            pairs='matching answer activation minus non-matching answer activation for the same question orientation',
            averaging='FP32 mean of paired differences, equal weight per source orientation',
            extraction='decoder block output at completed answer-letter token; no appended EOS',
            final_test_used=False),
        layers=dict(candidates={'llama2':[13],'llama3b':[7,9,11],'phi':[20,22,24],'qwen3':[16,18,20]},
            selection='Llama2 fixed validated block13. Other models select one shared block maximizing macro-concept mean validation P(+1)-P(-1), averaging original/swapped views within groups. Ties within 1e-12 prefer Phase-A anchor then lower block.',
            zero_based=True,hidden_state_index='block + 1',test_access='after prompt, norm protocol, and layer are frozen'),
        runtime=dict(dtype='float32',tf32=False,attention='eager',batch_size=16,padding='left',
            explicit_position_ids=True,use_cache=False,batch_parity_logit_atol=0.002,batch_parity_probability_atol=0.0001,
            placement='Llama2: two A5500 GPUs with the validated RCA static placement adapter; smaller models: one GPU'),
        prompt=dict(llama2='unchanged validated CAA [INST] question [/INST] (',
            other_models='first train-only semantic-gate passing protocol in fixed order',
            candidate_order=['legacy_parenthesis','letter_only','answer_prefill'],
            letter_only_instruction='Answer the multiple-choice question with exactly one uppercase letter, A or B. Do not add any explanation.',
            answer_prefill='The answer is ',
            gate_groups_per_concept=8,gate='On 64 train-only original/swapped views: at least 90% greedy next tokens are the unchanged bare A/B candidates; median A/B mass >= 0.1; every view has A/B mass > 1e-8. First passing candidate, never highest behavioral score.',
            qwen_enable_thinking=False,system_message='no explicit system message; retain native defaults',
            compatibility='Prompt adaptation is explicit and model-specific; the frozen bare-A/B normalized probability formula, target labels, and +/-1 conditions never change.'),
        evaluation=dict(alphas=[-1,0,1],zero='true no-hook',orientations='original and actual text/label swap',
            primary_metric='mean A/B-normalized target probability; average views within canonical group first',
            diagnostic_metrics=['full-vocabulary probabilities','logit margins','choice-order gaps','floor/ceiling','activation displacement'],
            bootstrap=dict(unit='canonical question group',paired=True,draws=2000,seed=42,ci=0.95),
            test='full existing test split; shared questions across all four models'),
        generation=dict(conditions=[-1,0,1],concepts='all four',max_new_tokens=64,do_sample=False,
            purpose='small open-ended coherence sanity check; no LLM judge and no replacement of the primary metric'),
        execution=dict(order='Llama2 validation and full test must pass before any other model inference',
            curveball=False,angular=False,github_push=False,overwrite_historical=False),
        original_reference=dict(code=str(ROOT/'experiments/caa_reproduction'),
            linear_eval_sha256=sha(ROOT/'experiments/caa_reproduction/linear_eval.py'),
            regression='reuse successful original50 regression and 20261027 RCA; run small cached own-data numerical parity checks only'))
    save(HERE/'experiment_config.json',cfg);save(RESULT/'experiment_config.json',cfg)
    save(RESULT/'data_audit.json',dict(status='passed',cross_concept_split_conflicts=0,counts=counts,hashes=hashes,
        limitation='Exact canonical grouping includes literal swaps and whitespace normalization, not semantic paraphrases. Split files were inspected for integrity before any outcomes.'))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
