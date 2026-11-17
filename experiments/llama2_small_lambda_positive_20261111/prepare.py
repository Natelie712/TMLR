"""Freeze the experiment; reuse or fit training-only representations on CPU."""
import json, os, time
import joblib, numpy as np, torch
from study_common import *

def freeze():
    old=load(ROOT/'results/qwen_original_style_20261105/experiment_config.json')
    src={'llama2':str(ROOT/'results/caa_official_seven_behaviors_four_models/staged_execution_20261030/llama2_three-method-comparison_16614779')}
    inputs=list(old['frozen_inputs'])
    inputs += [str(p.relative_to(ROOT)) for p in HERE.glob('*.py')]
    inputs += [str(p.relative_to(ROOT)) for p in HERE.glob('*.sh')]
    inputs += [str((OUT/'protocol.md').relative_to(ROOT))]
    for key,folder in src.items():
        inputs += [str((Path(folder)/n).relative_to(ROOT)) for n in ['manifest.json','training_prompts.jsonl','extraction_index_checks.json']+[f'train_activations_{c}.pt' for c in NAMES]]
    inputs += ['results/qwen_original_style_20261105/summary.csv','results/qwen_original_style_20261105/experiment_config.json']
    cfg=dict(models={k:stage.MODELS[k] for k in src},concepts=list(NAMES),weights=WEIGHTS,partitions=['validation','fit','benchmark'],positive_only=True,
        scorer='Fixed all-option normalized target-label probability',prompt_variant='legacy_parenthesis',scope='last real token',
        per_token_displacement_matching=False,precision='float32 LLM, float64 KPCA, TF32 disabled',
        split_manifest='results/qwen_quadratic_attenuation_20261103/split_manifest.json',counts=old['counts'],activation_sources=src,
        
        strength_rule='Calibrate lambda=1 once against the fit-only seven-behavior calibrated Linear vector RMS, using prefix-boundary-to-end activations from the first 32 sorted fit groups. Fix this native positive step across all lambda values. Evaluate only at the last real token. No per-token displacement matching or behavioral strength tuning.',
        selection_rule='After all validation results complete, choose the highest mean positive delta among lambda 0,.001,.005,.01,.05,.1,.25,.5; ties within 1e-6 pp choose the smaller lambda. Lambda=1 is a reference only. Freeze choices and hashes before official-test inference. Evaluate only Linear, lambda=0, selected lambda and lambda=1 on test.',
        selection_eligible_weights=WEIGHTS[:-1],
        budget='User removed prior 90-minute cap. Use one adequate GPU with checkpointed execution.',
        formal_test_execution=True,exploratory_fixed_benchmark=True,
        bootstrap=dict(resamples=10000,unit='source group',point_estimate='original-row mean',seed_rule='20261104 + concept_index*3 + split_index (fit=0, validation=1, benchmark=2)'),
        frozen_inputs={p:sha(ROOT/p) for p in sorted(set(inputs))})
    path=OUT/'experiment_config.json'
    if path.exists():assert load(path)==cfg,'Existing freeze differs; do not overwrite'
    else:save(path,cfg)
    # Check source-group separation including the benchmark without evaluating it.
    for c in NAMES:
        gs={p:{r['group_id'] for r in SPLIT if r['concept']==c and r['partition']==p} for p in ['fit','validation','benchmark']}
        assert not (gs['fit']&gs['validation'] or gs['fit']&gs['benchmark'] or gs['validation']&gs['benchmark'])
    save(OUT/'split_checks.json',dict(passed=True,split_sha256=sha(ROOT/cfg['split_manifest']),fit_questions=3906,validation_questions=912,official_test_evaluated=False))
    return cfg

def prepare():
    torch.set_num_threads(2);cfg=freeze();checks=[]
    for key,folder in cfg['activation_sources'].items():
        folder=Path(folder);target=OUT/'fits'/key;target.mkdir(parents=True,exist_ok=True)
        raw={};caches={}
        meta=load(folder/'manifest.json');assert meta['model']==cfg['models'][key]
        for c in NAMES:
            cache=torch.load(folder/f'train_activations_{c}.pt',weights_only=True)
            assert cache['source_indices']==list(range(len(data(c,'generate'))))
            idx=[r['index'] for r in SPLIT if r['concept']==c and r['partition']=='fit']
            assert len(idx)==cfg['counts'][c]['fit']['rows']
            pos,neg=cache['positive'][idx],cache['negative'][idx];caches[c]=(pos,neg)
            raw[c]=(pos-neg).mean(0)
        norm=torch.stack([v.norm() for v in raw.values()]).mean()
        vectors={c:v*norm/v.norm() for c,v in raw.items()}
        vp=target/'linear_vectors.pt'
        if not vp.exists():torch.save(dict(raw=raw,normalized=vectors,provenance='fit groups only; per-layer seven-behavior mean raw norm'),vp)
        else:
            saved=torch.load(vp,weights_only=True);assert all(torch.equal(saved['normalized'][c],vectors[c]) for c in NAMES)
        save(target/'linear_provenance.json',[dict(concept=c,pairs=len(caches[c][0]),raw_norm=float(raw[c].norm()),calibrated_norm=float(vectors[c].norm())) for c in NAMES])
        for c,(pos,neg) in caches.items():
            x=torch.cat([pos,neg]).numpy();labels=np.r_[np.ones(len(pos)),-np.ones(len(neg))]
            for w in WEIGHTS:
                path=fit_path(key,c,w);source=None;t=time.time()
                if path.exists():cb=AttenuatedCurveball.restore(joblib.load(path));source='resumed verified checkpoint'
                else:
                    cb=AttenuatedCurveball(x,labels,w);source='new training-only fit'
                np.testing.assert_allclose(cb.mean,x.astype('float64').mean(0),rtol=0,atol=1e-10)
                np.testing.assert_array_equal(cb.fit.X_fit_,x.astype('float64')-cb.mean)
                h=torch.from_numpy(x[:8]).double();z=cb.project(h)
                zp=cb.fit.transform(x[:8].astype('float64')-cb.mean)
                np.testing.assert_allclose(z.numpy(),zp,rtol=1e-7,atol=1e-7)
                np.testing.assert_allclose(cb.inverse(z).numpy(),cb.fit.inverse_transform(zp),rtol=1e-6,atol=1e-5)
                assert torch.equal(cb.intervention(h,0),h)
                explicit=h+cb.inverse(z+0*cb.direction)-cb.inverse(z)
                assert float((explicit-h).abs().max())<1e-9
                zall=cb.fit.transform(x.astype('float64')-cb.mean);d=zall[:len(pos)].mean(0)-zall[len(pos):].mean(0);d/=np.linalg.norm(d)
                np.testing.assert_allclose(cb.direction.numpy(),d,rtol=1e-6,atol=1e-7)
                expected=1+2*(.01*h@h.T)+w*(.01*h@h.T)**2
                np.testing.assert_allclose(cb.kernel(h,h).numpy(),expected.numpy(),rtol=1e-12,atol=1e-9)
                if not path.exists():joblib.dump(cb.state(),path)
                check=dict(cb.fit_diagnostics,model=key,concept=c,weight=w,source=source,passed=True,elapsed_seconds=time.time()-t,fit_sha256=sha(path))
                checks.append(check);save(OUT/'cpu_checks.json',dict(passed=True,completed=len(checks),expected=63,checks=checks))
                print(json.dumps(dict(stage='fit_ready',model=key,concept=c,weight=w,seconds=round(time.time()-t,2))),flush=True)
        save(target/'freeze.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in target.glob('*') if p.name!='freeze.json'}))
    verify();save(OUT/'preparation_complete.json',dict(passed=True,fits=63,model_weights_updated=False))

if __name__=='__main__':prepare()
