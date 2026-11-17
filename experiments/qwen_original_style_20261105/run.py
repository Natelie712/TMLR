"""Frozen last-token, unmatched Curveball evaluation. Never modifies old artifacts."""
import argparse, json, os, signal, sys, time, traceback, shutil
from pathlib import Path
import joblib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/qwen_quadratic_attenuation_20261103'))
# Load the prior module explicitly: this file is also named run.py.
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_attenuation_runner',ROOT/'experiments/qwen_quadratic_attenuation_20261103/run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from weighted import AttenuatedCurveball
from scoring import score
OUT=ROOT/'results/qwen_original_style_20261105'
CFG=json.loads((OUT/'experiment_config.json').read_text())
save=prior.save

class Experiment(prior.stage.Experiment):
    def __init__(self,shard):
        old=prior.stage.RESULT;prior.stage.RESULT=OUT
        super().__init__('qwen3',f'original-style-{shard}');prior.stage.RESULT=old
        self.shard=shard;self.concepts=CFG['shards'][str(shard)];self.variant='assistant_answer_parenthesis'
        self.vectors=torch.load(ROOT/CFG['linear_vectors'],weights_only=True)['normalized']
        self.cbs={};self.steps=CFG['native_steps'];self.completed=[];self.count=0;self.bs=16;self.reference={}
        self.manifest.update(protocol='last_token_unmatched',paper_exact_reproduction=False,config_sha256=prior.stage.tc.sha(OUT/'experiment_config.json'))
        for p in [Path(__file__),OUT/'experiment_config.json',OUT/'protocol.md']:
            shutil.copy2(p,self.out/'sources'/p.name)
        self.persist('initialized')

    def persist(self,status):
        self.manifest.update(status=status,completed=self.completed,question_condition_rows=self.count,elapsed_seconds=time.time()-self.start)
        save(self.out/'manifest.json',self.manifest)
        save(OUT/f'shard_{self.shard}_location.json',dict(path=str(self.out),status=status))

    def subset(self,c,partition):return prior.Ablation.subset(self,c,partition)

    def restore(self):
        records=[]
        for c in self.concepts:
            self.cbs[c]={}
            for w in [0.,.5,1.]:
                p=ROOT/CFG['fits'][c][str(w)];cb=AttenuatedCurveball.restore(joblib.load(p),'cpu')
                assert abs(float(cb.direction.norm())-1)<1e-10
                self.cbs[c][w]=cb
                records.append(dict(cb.fit_diagnostics,concept=c,weight=w,source=str(p),sha256=prior.stage.tc.sha(p)))
        save(self.out/'fit_diagnostics.json',records)
        save(self.out/'native_parameters.json',self.steps)

    def condition(self,tasks,method,sign,matched=False,step_override=None):
        c=tasks[0]['concept'];obs=[];calls=[];handle=None
        cb=self.cbs[c][float(method.split('_')[1])] if method.startswith('lambda_') else None
        step=self.steps[c][str(sign)] if cb is not None and sign else 0.
        if step_override is not None:step=step_override
        v=self.vectors[c].cuda();budget=float(self.vectors[c].double().norm())
        def hook(module,args,kwargs,out):
            calls.append(1);h=out[0] if isinstance(out,tuple) else out
            before=h[:,-1,:].clone();hd=before.double();position=kwargs['position_ids'][:,-1]
            assert position.cpu().tolist()==[len(t['ids'])-1 for t in tasks]
            if cb is None:after=(before+sign*v.to(before))
            else:
                after=cb.intervention(before,step)
                if matched and sign:
                    delta=after.double()-hd;norm=delta.norm(dim=-1,keepdim=True)
                    assert norm.min()>1e-10
                    after=(hd+budget*delta/norm).to(before.dtype)
            delta=after.double()-hd
            assert torch.isfinite(after).all()
            if cb is not None:
                z=cb.project(before);err=(cb.inverse(z)+cb.center-hd).norm(dim=-1)
            else:err=torch.zeros(len(tasks),device=h.device,dtype=torch.float64)
            linear=(before+sign*v.to(before)).double()-hd
            cosine=torch.nn.functional.cosine_similarity(delta,linear,dim=-1)
            for j,t in enumerate(tasks):
                if t['index'] in self.reference:assert torch.equal(self.reference[t['index']],hd[j].cpu())
                else:self.reference[t['index']]=hd[j].cpu()
                obs.append(dict(positions=[int(position[j])],actual_displacement=[float(delta[j].norm())],activation_norm=[float(hd[j].norm())],
                    relative_displacement=[float(delta[j].norm()/hd[j].norm())],cosine_to_actual_signed_linear=[float(cosine[j])],
                    reconstruction_l2=[float(err[j])] if cb is not None else [],relative_reconstruction_l2=[float(err[j]/hd[j].norm())] if cb is not None else []))
            # Clone and replace exactly the last token, as in the original hook.
            changed=h.clone();changed[:,-1,:]=after
            assert torch.equal(changed[:,:-1,:],h[:,:-1,:])
            return (changed,)+out[1:] if isinstance(out,tuple) else changed
        if method!='No-hook':handle=self.block.register_forward_hook(hook,with_kwargs=True)
        try:rows,_=self.evaluate(tasks,0)
        finally:
            if handle:handle.remove()
        if handle:assert len(calls)==1 and len(obs)==len(tasks)
        for i,(r,t) in enumerate(zip(rows,tasks)):
            r.update(method=method,alpha=sign,condition='last_token_matched_bridge' if matched else 'last_token_unmatched',native_parameter=step if cb else sign,
                group_id=t['group_id'],partition=t['partition'],intervention_triggered=bool(sign),
                displacement=obs[i] if handle else dict(positions=[],actual_displacement=[0.],activation_norm=[],relative_displacement=[0.]),
                execution_provenance='new original-style GPU job '+str(self.manifest['job_id']))
            assert np.isfinite(list(r['candidate_logits'].values())).all()
        return rows

    def checks(self):
        c=self.concepts[0];tasks=sorted(self.subset(c,'fit'),key=lambda t:len(t['ids']))[:16]
        for cb in self.cbs[c].values():cb.to('cuda')
        outputs={};checks=[]
        for bs in [4,16]:
            self.bs=bs;rows=[];self.reference={}
            for offset in range(0,len(tasks),bs):
                batch=tasks[offset:offset+bs]
                rows+=self.condition(batch,'No-hook',0)
                rows+=self.condition(batch,'Linear',1)
                rows+=self.condition(batch,'lambda_1.0',1)
            outputs[bs]={(r['index'],r['method']):r['normalized_target_probability'] for r in rows}
        error=max(abs(outputs[4][k]-outputs[16][k]) for k in outputs[4]);self.bs=16 if error<1e-4 else 4
        # Zero-step hooks must equal a true hook-free forward.
        ts=tasks[:4];self.reference={};base=self.condition(ts,'No-hook',0)
        for w in [0.,.5,1.]:
            zero=self.condition(ts,f'lambda_{w}',0,step_override=0.)
            e=max(abs(a['normalized_target_probability']-b['normalized_target_probability']) for a,b in zip(base,zero))
            assert e==0, e
            cb=self.cbs[c][w];h=torch.stack([self.reference[t['index']] for t in ts]).cuda()
            z=cb.project(h)
            np.testing.assert_allclose(z.cpu(),cb.fit.transform(h.cpu().numpy()-cb.mean),rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(cb.inverse(z).cpu(),cb.fit.inverse_transform(z.cpu().numpy()),rtol=1e-6,atol=1e-5)
            checks.append(dict(weight=w,zero_hook_probability_error=e))
        # Actual hidden-state index equals decoder block output on this model.
        captured=[]
        def observe(module,args,out):captured.append((out[0] if isinstance(out,tuple) else out)[:,-1,:].clone())
        handle=self.block.register_forward_hook(observe)
        try:ret,_=self.forward(ts,hidden=True)
        finally:handle.remove()
        assert torch.equal(captured[0],ret.hidden_states[19][:,-1,:])
        save(self.out/'correctness_checks.json',dict(passed=True,training_only=True,block=18,hidden_state_index=19,batch4_vs16_max_probability_error=error,
            selected_batch_size=self.bs,zero_hooks=checks,last_token_only=True,true_nohook_baseline=True))
        for cb in self.cbs[c].values():cb.to('cpu')
        self.reference={}

    def batches(self,tasks):
        batch=[]
        for t in sorted(tasks,key=lambda t:len(t['ids'])):
            limit=min(self.bs,4 if len(t['ids'])>512 else 8 if len(t['ids'])>256 else 16)
            if batch and len(batch)>=limit:yield batch;batch=[]
            batch.append(t)
        if batch:yield batch

    def evaluate_partition(self,c,partition):
        tasks=self.subset(c,partition);save(self.out/f'prompts_{c}_{partition}.json',tasks)
        path=self.out/f'rows_{c}_{partition}.jsonl';assert not path.exists()
        bridge=self.out/f'bridge_{c}_validation.jsonl'
        with path.open('w') as f:
            bridge_file=bridge.open('w') if partition=='validation' else None
            try:
                done=0
                for batch in self.batches(tasks):
                    self.reference={};rows=self.condition(batch,'No-hook',0)
                    for method in ['Linear','lambda_0.0','lambda_0.5','lambda_1.0']:
                        for a in [-1,1]:rows+=self.condition(batch,method,a)
                    assert len(rows)==9*len(batch)
                    f.write(json.dumps(dict(rows=rows))+'\n');f.flush();os.fsync(f.fileno());self.count+=len(rows)
                    if bridge_file:
                        bridge_rows=[]
                        for w in [0.,.5,1.]:
                            for a in [-1,1]:bridge_rows+=self.condition(batch,f'lambda_{w}',a,matched=True)
                        bridge_file.write(json.dumps(dict(rows=bridge_rows))+'\n');bridge_file.flush();os.fsync(bridge_file.fileno())
                    done+=len(batch)
                    if done%128<len(batch):self.progress('scoring',concept=c,partition=partition,questions_done=done,questions_total=len(tasks));self.persist('running')
            finally:
                if bridge_file:bridge_file.close()
        self.completed.append(dict(concept=c,partition=partition,questions=len(tasks)));self.persist('running')
        self.progress('partition_complete',concept=c,partition=partition,questions=len(tasks))

    def coherence(self,c):
        tasks=self.subset(c,'validation');by_id={t['index']:t for t in tasks}
        chosen=[by_id[i] for i in CFG['coherence_question_ids'][c]]
        messages=['Consider the following question and options. Explain your answer in a short paragraph.\n\n'+t['question'] for t in chosen]
        rendered=[self.tok.apply_chat_template([dict(role='user',content=q)],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in messages]
        saved=[];self.tok.padding_side='left'
        if self.tok.pad_token_id is None:self.tok.pad_token_id=self.tok.eos_token_id
        for method,a in [('No-hook',0)]+[(m,a) for m in ['Linear','lambda_0.0','lambda_0.5','lambda_1.0'] for a in [-1,1]]:
            cb=self.cbs[c][float(method.split('_')[1])] if method.startswith('lambda_') else None
            step=self.steps[c][str(a)] if cb else a;updates=[];handle=None
            def hook(module,args,out):
                h=out[0] if isinstance(out,tuple) else out;b=h[:,-1,:].clone()
                after=cb.intervention(b,step) if cb else b+a*self.vectors[c].to(b)
                assert torch.isfinite(after).all();updates.append((after.double()-b.double()).norm(dim=-1).cpu().tolist())
                h=h.clone();h[:,-1,:]=after
                return (h,)+out[1:] if isinstance(out,tuple) else h
            if a:handle=self.block.register_forward_hook(hook)
            encoded=self.tok(rendered,padding=True,return_tensors='pt').to('cuda')
            try:output=self.model.generate(**encoded,max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
            finally:
                if handle:handle.remove()
            tokens=output[:,encoded.input_ids.shape[1]:]
            for j,t in enumerate(chosen):
                ids=tokens[j].cpu().tolist();eos=self.tok.eos_token_id
                if eos in ids:ids=ids[:ids.index(eos)]
                txt=self.tok.decode(ids,skip_special_tokens=True);words=txt.lower().split();grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
                saved.append(dict(concept=c,index=t['index'],group_id=t['group_id'],method=method,alpha=a,native_parameter=step,question=messages[j],rendered_prompt=rendered[j],
                    response=txt,generated_tokens=ids,hit_length_cap=len(ids)>=96,empty_response=not bool(txt.strip()),
                    repeated_fourgram_fraction=0. if not grams else 1-len(set(grams))/len(grams),
                    displacement=[v[j] for v in updates],intervention_scope='last token on every generation forward; KV cache enabled',
                    status='automated screens only; semantic coherence requires reading the response'))
        save(self.out/f'coherence_{c}.json',saved);self.progress('coherence_complete',concept=c,outputs=len(saved))

    def run(self):
        self.load();self.restore();self.checks();self.persist('checks_passed')
        for c in self.concepts:
            for cb in self.cbs[c].values():cb.to('cuda')
            for partition in ['fit','validation','benchmark']:self.evaluate_partition(c,partition)
            self.coherence(c)
            for cb in self.cbs[c].values():cb.to('cpu')
        self.persist('complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True,choices=[0,1]);args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for p,digest in CFG['frozen_inputs'].items():assert prior.stage.tc.sha(ROOT/p)==digest,p
    ex=Experiment(args.shard)
    def stop(sig,frame):ex.persist('interrupted_checkpoint_saved');raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise

if __name__=='__main__':main()
