"""Fit-only Qwen attenuation experiment; validation selection precedes benchmark."""
import sys,json,math,os,traceback,hashlib,shutil,re,time
from pathlib import Path
import numpy as np
import torch
import joblib
from weighted import AttenuatedCurveball,ROOT
BASE=ROOT/'experiments/caa_official_seven_behaviors_four_models'
sys.path.insert(0,str(BASE))
import run_stage_b as stage
from runtime_b import task,data
from scoring import parse_options
OUT=ROOT/'results/qwen_quadratic_attenuation_20261103'
CFG=json.loads((OUT/'experiment_config.json').read_text())
SOURCE=Path(CFG['source_run'])
SPLIT=json.loads((OUT/'split_manifest.json').read_text())
CONCEPTS=CFG['concepts']
LABEL=lambda w:'lambda_'+str(float(w))
def save(path,obj):stage.tc.save(path,obj)
def load(path):return json.loads(path.read_text())

class Ablation(stage.Experiment):
    def __init__(self,mode):
        old=stage.RESULT;stage.RESULT=OUT
        super().__init__('qwen3','attenuation-'+mode);stage.RESULT=old
        self.variant='assistant_answer_parenthesis';self.mode=mode
        assert load(OUT/'kernel_compatibility.json')['passed']
        assert stage.tc.sha(OUT/'split_manifest.json')==CFG['split_sha256']
        for n,h in CFG['source_hashes'].items():assert stage.tc.sha(ROOT/n)==h,n
        for p in [Path(__file__),Path(__file__).with_name('weighted.py'),OUT/'experiment_config.json',OUT/'split_manifest.json']:
            shutil.copy2(p,self.out/'sources'/p.name)
        self.meta={(r['concept'],r['split'],r['index']):r for r in SPLIT}

    def subset(self,c,partition):
        split='test' if partition=='benchmark' else 'generate';source=data(c,split)
        return [dict(task(self.tok,'qwen3',c,r['index'],source[r['index']],self.variant,split),group_id=r['group_id'],partition=partition)
                for r in SPLIT if r['concept']==c and r['partition']==partition]

    def regression(self):
        sys.path.insert(0,str(ROOT/'experiments/caa_postrun_validation_20261102'))
        from replay import load_curveball
        old={(r['concept'],r['index'],r['method'],r['alpha']):r for r in load(SOURCE/'comparison_rows.json')}
        oldv=torch.load(SOURCE/'newly_extracted_vectors.pt',weights_only=True)['normalized']
        fits={r['concept']:r for r in load(SOURCE/'method_fit_diagnostics.json')}
        checks=[]
        for c in CONCEPTS:
            tasks=self.subset(c,'benchmark')[:3]
            for a in [0,-1,1]:
                rr,_=self.evaluate(tasks,a,oldv)
                checks += [dict(concept=c,index=r['index'],method='Linear',alpha=a,error=abs(r['normalized_target_probability']-old[c,r['index'],'Linear',a]['normalized_target_probability'])) for r in rr]
            cb=load_curveball(SOURCE/f'curveball_fit_{c}.joblib','cuda')
            for a in [-1,1]:
                rr=self.cb_rows(tasks,cb,fits[c]['strengths'][str(a)]['native_parameter'],a,'historical_Curveball','fixed')
                checks += [dict(concept=c,index=r['index'],method='Curveball',alpha=a,error=abs(r['normalized_target_probability']-old[c,r['index'],'Curveball',a]['normalized_target_probability'])) for r in rr]
            cb.to('cpu')
        assert max(r['error'] for r in checks)<1e-4
        save(self.out/'historical_regression.json',checks);self.progress('historical_regression_passed')

    def cb_rows(self,tasks,cb,parameter,alpha,method,policy):
        result=[]
        for batch in stage.tc.batches(tasks,self.bs):
            obs=[]
            def hook(module,args,kwargs,out):
                h=out[0] if isinstance(out,tuple) else out
                bounds=torch.tensor([[t['decision']['boundary']] for t in batch],device=h.device)
                mask=kwargs['position_ids']>=bounds;before=h[mask].clone()
                after=cb.intervention(before,parameter);delta=after-before;h[mask]=after
                assert torch.isfinite(after).all()
                z=cb.project(before);err=(cb.inverse(z)+cb.center-before.double()).norm(dim=-1)
                c=batch[0]['concept'];v=self.vectors[c].to(delta)
                cosine=torch.nn.functional.cosine_similarity(delta.double(),(alpha*v).double()[None,:],dim=-1)
                cursor=0
                for i,t in enumerate(batch):
                    n=int(mask[i].sum());s=slice(cursor,cursor+n);cursor+=n
                    obs.append(dict(actual_displacement=delta[s].norm(dim=-1).cpu().tolist(),activation_norm=before[s].norm(dim=-1).cpu().tolist(),
                        relative_displacement=(delta[s].norm(dim=-1)/before[s].norm(dim=-1)).cpu().tolist(),
                        cosine_to_signed_linear=cosine[s].cpu().tolist(),reconstruction_l2=err[s].cpu().tolist(),
                        relative_reconstruction_l2=(err[s]/before[s].double().norm(dim=-1)).cpu().tolist(),positions=kwargs['position_ids'][i,mask[i]].cpu().tolist()))
                return out
            handle=self.block.register_forward_hook(hook,with_kwargs=True)
            try:rr,_=self.evaluate(batch,0)
            finally:handle.remove()
            by={t['index']:(o,t) for t,o in zip(batch,obs)}
            for r in rr:
                o,t=by[r['index']];r.update(method=method,alpha=alpha,strength_policy=policy,native_parameter=parameter,
                    displacement=o,intervention_triggered=True,group_id=t['group_id'],partition=t['partition'])
            result.extend(rr)
        return result

    def linear_rows(self,tasks,alpha):
        rr,_=self.evaluate(tasks,alpha,self.vectors);by={t['index']:t for t in tasks}
        for r in rr:r.update(method='Linear' if alpha else 'No-hook',strength_policy='fixed',group_id=by[r['index']]['group_id'],partition=by[r['index']]['partition'])
        return rr

    def fit(self):
        raw={}
        for c in stage.CONCEPTS:
            cached=torch.load(SOURCE/f'train_activations_{c}.pt',weights_only=True)
            idx=[r['index'] for r in SPLIT if r['concept']==c and r['partition']=='fit']
            raw[c]=(cached['positive'][idx]-cached['negative'][idx]).mean(0)
        norm=torch.stack([raw[c].norm() for c in stage.CONCEPTS]).mean()
        self.vectors={c:v*norm/v.norm() for c,v in raw.items()}
        torch.save(dict(raw=raw,normalized=self.vectors,source='fit groups only, cached independent per-example activations'),self.out/'fit_only_linear_vectors.pt')
        self.regression()
        fitmeta=[];self.steps={}
        for c in CONCEPTS:
            cached=torch.load(SOURCE/f'train_activations_{c}.pt',weights_only=True)
            idx=[r['index'] for r in SPLIT if r['concept']==c and r['partition']=='fit']
            x=torch.cat([cached['positive'][idx],cached['negative'][idx]]).numpy();labels=np.r_[np.ones(len(idx)),-np.ones(len(idx))]
            bygroup={}
            for t in self.subset(c,'fit'):bygroup.setdefault(t['group_id'],t)
            probes=[];probetasks=[bygroup[g] for g in sorted(bygroup)[:32]]
            for t in probetasks:
                def capture(module,args,kwargs,out):
                    h=out[0] if isinstance(out,tuple) else out
                    probes.append(h[kwargs['position_ids']>=t['decision']['boundary']].clone())
                handle=self.block.register_forward_hook(capture,with_kwargs=True)
                try:self.forward([t])
                finally:handle.remove()
            probes=torch.cat(probes);torch.save(dict(activations=probes.cpu(),groups=[t['group_id'] for t in probetasks]),self.out/f'fit_probes_{c}.pt')
            self.steps[c]={}
            for w in [1.,0.,.1,.25,.5,.75]:
                cb=AttenuatedCurveball(x,labels,w,device='cuda')
                check=probes[:8];z=cb.project(check).cpu().numpy()
                np.testing.assert_allclose(z,cb.fit.transform(check.cpu().double().numpy()-cb.mean),rtol=1e-6,atol=1e-6)
                inverse=cb.inverse(cb.project(check)).cpu().numpy()
                np.testing.assert_allclose(inverse,cb.fit.inverse_transform(z),rtol=1e-6,atol=1e-5)
                assert torch.equal(cb.intervention(check,0),check)
                matched={str(a):cb.match_strength(probes,float(self.vectors[c].norm()),a) for a in [-1,1]}
                self.steps[c][LABEL(w)]=matched
                fitmeta.append(dict(concept=c,**cb.fit_diagnostics,matched=matched))
                joblib.dump(cb.state(),self.out/f'fit_{c}_{LABEL(w)}.joblib');cb.to('cpu')
                save(self.out/'fit_diagnostics.json',fitmeta);save(self.out/'strength_parameters.json',self.steps)
                self.progress('fit_complete',concept=c,weight=w)

    def evaluate_partition(self,partition):
        allrows=[];prompts=[]
        for c in CONCEPTS:
            tasks=self.subset(c,partition);prompts+=tasks
            for a in [0,-1,1]:allrows+=self.linear_rows(tasks,a)
            for w in CFG['weights']:
                name=LABEL(w);cb=AttenuatedCurveball.restore(joblib.load(self.fitroot/f'fit_{c}_{name}.joblib'),'cuda')
                for policy in ['fixed','matched']:
                    for a in [-1,1]:
                        param=self.steps[c][LABEL(1.) if policy=='fixed' else name][str(a)]['native_parameter']
                        allrows+=self.cb_rows(tasks,cb,param,a,name,policy)
                cb.to('cpu');save(self.out/f'{partition}_rows.json',allrows)
                self.progress('evaluated',partition=partition,concept=c,weight=w)
        save(self.out/f'{partition}_prompts.json',prompts)
        return allrows

    def coherence(self):
        outputs=[]
        for c in CONCEPTS:
            groups={}
            for t in self.subset(c,'validation'):groups.setdefault(t['group_id'],t)
            selected=[groups[g] for g in sorted(groups)[:4]]
            for name in ['No-hook','Linear']+[LABEL(w) for w in CFG['weights']]:
                cb=None
                if name.startswith('lambda_'):cb=AttenuatedCurveball.restore(joblib.load(self.fitroot/f'fit_{c}_{name}.joblib'),'cuda')
                for a in ([0] if name=='No-hook' else [-1,1]):
                    tasks=[]
                    for t in selected:
                        question=t['question'];m=re.search(r'^[ \t]*\([A-Z]\)',question,re.M)
                        stem=question[:m.start()].replace('Choices:','').strip()
                        prompt=stem+'\n\nPlease answer in plain English and briefly explain your reasoning.'
                        d=stage.tc.legacy.prompt(self.tok,'qwen3',prompt,answer='')
                        tasks.append(dict(ids=d['input_ids'],decision=d,group_id=t['group_id'],index=t['index'],prompt=d['rendered']))
                    ids,mask,pos,pads=stage.tc.tensors(self.tok,tasks)
                    bounds=torch.tensor([[t['decision']['boundary']] for t in tasks],device='cuda')
                    calls=[0]
                    def hook(module,args,kwargs,out):
                        h=out[0] if isinstance(out,tuple) else out;positions=kwargs['position_ids'];scope=positions>=bounds
                        if name=='Linear':h[scope]+=a*self.vectors[c].to(h)
                        else:h[scope]=cb.intervention(h[scope],self.steps[c][LABEL(1.)][str(a)]['native_parameter'])
                        calls[0]+=1
                        return out
                    handle=self.block.register_forward_hook(hook,with_kwargs=True) if a else None
                    try:
                        generated=self.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=128,do_sample=False,pad_token_id=self.tok.eos_token_id)
                    finally:
                        if handle:handle.remove()
                    for j,t in enumerate(tasks):
                        seq=generated[j,ids.shape[1]:].cpu().tolist()
                        eos=self.tok.eos_token_id
                        if eos in seq:seq=seq[:seq.index(eos)+1]
                        text=self.tok.decode(seq,skip_special_tokens=True);words=re.findall(r"\b[\w']+\b",text.lower())
                        grams=list(zip(words,words[1:],words[2:]));repeat=1-len(set(grams))/len(grams) if grams else 0.
                        failure=not text.strip() or len(words)<4 or repeat>.20
                        outputs.append(dict(concept=c,method=name,alpha=a,index=t['index'],group_id=t['group_id'],prompt=t['prompt'],text=text,
                            generated_ids=seq,token_count=len(seq),word_count=len(words),trigram_repetition=repeat,
                            empty_output=not bool(text.strip()),hit_length_limit=len(seq)>=128,automatic_failure=failure,hook_calls=calls[0],
                            scope='same inclusive delimiter and subsequent tokens, including generation; open-ended assistant prefill is empty'))
                    save(self.out/'coherence_generations.json',outputs)
                if cb:cb.to('cpu')
                self.progress('coherence_complete',concept=c,method=name)

    def run(self):
        self.load()
        if self.mode=='validation':
            self.fit();self.fitroot=self.out
            self.evaluate_partition('validation');self.coherence()
            self.manifest.update(status='validation_complete',new_candidate_benchmark_evaluated=False,historical_regression_benchmark_questions=9)
            save(OUT/'validation_run.json',dict(path=str(self.out)))
        else:
            self.fitroot=Path(load(OUT/'validation_run.json')['path'])
            selection=load(OUT/'selection.json');assert selection['frozen_before_benchmark']
            self.steps=load(self.fitroot/'strength_parameters.json')
            self.vectors=torch.load(self.fitroot/'fit_only_linear_vectors.pt',weights_only=True)['normalized']
            shutil.copy2(OUT/'selection.json',self.out/'sources/selection.json')
            self.evaluate_partition('benchmark')
            self.manifest.update(status='benchmark_complete',selection_sha256=stage.tc.sha(OUT/'selection.json'))
            save(OUT/'benchmark_run.json',dict(path=str(self.out)))
        save(self.out/'manifest.json',self.manifest)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','benchmark']);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ex=Ablation(a.mode)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest.update(status='failed',error=traceback.format_exc());save(ex.out/'manifest.json',ex.manifest);raise
