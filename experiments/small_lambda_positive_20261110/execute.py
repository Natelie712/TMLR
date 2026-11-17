"""Positive-only, checkpointed evaluation; never evaluates the official test."""
import argparse, os, signal, time, traceback
import joblib, numpy as np, torch
from study_common import *

class Experiment(original.Experiment):
    def __init__(self,key,partition):
        # Use the validated base initialization, not the Qwen-specific initializer.
        previous=stage.RESULT;stage.RESULT=OUT
        try:stage.Experiment.__init__(self,key,f'small-lambda-{partition}-{time.time_ns()}')
        finally:stage.RESULT=previous
        self.partition=partition;self.concepts=list(NAMES);self.variant='assistant_answer_parenthesis'
        self.cfg=config();self.vectors=torch.load(OUT/'fits'/key/'linear_vectors.pt',weights_only=True)['normalized']
        self.cbs={};self.steps={};self.completed=[];self.count=0;self.bs=4;self.reference={}
        self.manifest.update(protocol='positive-only last-token natural-magnitude small-lambda',formal_sweep=False,
            config_sha256=sha(OUT/'experiment_config.json'),official_test_evaluated=False)
        save(OUT/f'location_{key}_{partition}.json',dict(path=str(self.out)))
        self.persist('initialized')

    def persist(self,status):
        self.manifest.update(status=status,completed=self.completed,question_condition_rows=self.count,elapsed_seconds=time.time()-self.start)
        save(self.out/'manifest.json',self.manifest)
        save(OUT/f'status_{self.key}_{self.partition}.json',dict(status=status,path=str(self.out),completed=self.completed,rows=self.count))

    def subset(self,c,partition):
        assert partition in ['fit','validation'],'Official test is disabled in this stage'
        source=data(c,'generate')
        return [dict(task(self.tok,self.key,c,r['index'],source[r['index']],self.variant,'generate'),group_id=r['group_id'],partition=partition)
                for r in SPLIT if r['concept']==c and r['partition']==partition]

    def restore(self):
        folder=OUT/'fits'/self.key
        for p,h in load(folder/'freeze.json')['files'].items():assert sha(ROOT/p)==h,p
        for c in self.concepts:self.cbs[c]={w:AttenuatedCurveball.restore(joblib.load(fit_path(self.key,c,w))) for w in WEIGHTS}
        params=folder/'positive_steps.json'
        if params.exists():
            self.steps=load(params)['steps']
            assert load(params)['fit_hashes']=={c:sha(fit_path(self.key,c,1.)) for c in self.concepts}
        elif self.key=='qwen3':
            self.steps={c:{'1':self.cfg['qwen_native_steps'][c]['1']} for c in self.concepts}
            save(params,dict(steps=self.steps,source='Unchanged prior fit-only Qwen strengths',fit_hashes={c:sha(fit_path(self.key,c,1.)) for c in self.concepts}))
        else:
            diagnostics=[]
            for c in self.concepts:
                by={}
                for t in self.subset(c,'fit'):by.setdefault(t['group_id'],t)
                selected=[by[g] for g in sorted(by)[:32]];captured=[]
                for t in selected:
                    def capture(module,args,kwargs,out):
                        h=out[0] if isinstance(out,tuple) else out
                        captured.append(h[kwargs['position_ids']>=t['decision']['boundary']].clone())
                    handle=self.block.register_forward_hook(capture,with_kwargs=True)
                    try:self.forward([t])
                    finally:handle.remove()
                probes=torch.cat(captured);cb=self.cbs[c][1.].to('cuda')
                result=cb.match_strength(probes,float(self.vectors[c].double().norm()),1)
                self.steps[c]={'1':result['native_parameter']}
                diagnostics.append(dict(concept=c,groups=[t['group_id'] for t in selected],**result))
                torch.save(dict(activations=probes.cpu(),groups=[t['group_id'] for t in selected]),folder/f'calibration_probes_{c}.pt')
                cb.to('cpu');self.progress('strength_frozen',concept=c,positive_step=self.steps[c]['1'])
            save(folder/'positive_step_calibration.json',diagnostics)
            save(params,dict(steps=self.steps,source='lambda1 fit-only calibration, frozen across all weights before outcome evaluation',fit_hashes={c:sha(fit_path(self.key,c,1.)) for c in self.concepts}))
        save(self.out/'native_parameters.json',self.steps)
        save(self.out/'representation_hashes.json',dict(fits={str(fit_path(self.key,c,w).relative_to(ROOT)):sha(fit_path(self.key,c,w)) for c in self.concepts for w in WEIGHTS},positive_steps_sha256=sha(params),linear_sha256=sha(folder/'linear_vectors.pt')))

    def checks(self):
        checks=[];cache_checks=[];token_rows=[]
        # Check cached extraction provenance for every behavior against a fresh forward.
        for c in self.concepts:
            t=sorted(self.subset(c,'fit'),key=lambda t:t['index'])[0]
            cache=torch.load(Path(self.cfg['activation_sources'][self.key])/f'train_activations_{c}.pt',weights_only=True)
            for sign,label in [('positive',t['target']),('negative',t['nonmatching'])]:
                completed=dict(t,ids=t['decision']['completion_ids'][label]);captured=[]
                def hook(module,args,out):
                    h=out[0] if isinstance(out,tuple) else out;captured.append(h[:,t['decision']['extraction_position'],:].clone())
                handle=self.block.register_forward_hook(hook)
                try:ret,_=self.forward([completed],hidden=True)
                finally:handle.remove()
                assert torch.equal(captured[0],ret.hidden_states[self.block_index+1][:,t['decision']['extraction_position'],:])
                ref=cache[sign][t['index']].double();delta=captured[0][0].cpu().double()-ref
                rel=float(delta.norm()/ref.norm());assert rel<1e-4,(c,sign,rel)
                cache_checks.append(dict(concept=c,sign=sign,index=t['index'],relative_activation_error=rel,max_absolute_error=float(delta.abs().max())))
            token_rows+=self.condition([t],'No-hook',0)
        save(self.out/'candidate_scoring_checks.json',dict(prompt_variant=self.variant,rows=token_rows,automatic_template_changes=False))
        # Check probability semantics before any interpretation, do not silently adapt scorer.
        assert all(r['all_option_probability_mass']>1e-8 for r in token_rows),'Candidate mass mismatch: report before adapting'
        c=self.concepts[0];ts=sorted(self.subset(c,'fit'),key=lambda t:len(t['ids']))[:4]
        for cb in self.cbs[c].values():cb.to('cuda')
        self.reference={};base=self.condition(ts,'No-hook',0)
        for w in WEIGHTS:
            self.reference={};zero=self.condition(ts,f'lambda_{w}',0,step_override=0.)
            err=max(abs(a['normalized_target_probability']-b['normalized_target_probability']) for a,b in zip(base,zero))
            assert err==0,err
            cb=self.cbs[c][w];h=torch.stack([self.reference[t['index']] for t in ts]).cuda();z=cb.project(h)
            np.testing.assert_allclose(z.cpu().numpy(),cb.fit.transform(h.cpu().numpy()-cb.mean),rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(cb.inverse(z).cpu().numpy(),cb.fit.inverse_transform(z.cpu().numpy()),rtol=1e-6,atol=1e-5)
            checks.append(dict(weight=w,zero_probability_error=err,gpu_projection_inverse_parity=True))
        batched={};single={}
        for method,sign in [('No-hook',0),('Linear',1),('lambda_1.0',1)]:
            self.reference={};rr=self.condition(ts,method,sign)
            batched.update({(r['index'],method):r['normalized_target_probability'] for r in rr})
            for t in ts:
                self.reference={};r=self.condition([t],method,sign)[0];single[t['index'],method]=r['normalized_target_probability']
        error=max(abs(batched[k]-single[k]) for k in batched)
        assert error<1e-4,error
        for cb in self.cbs[c].values():cb.to('cpu')
        self.reference={}
        save(self.out/'correctness_checks.json',dict(passed=True,training_only=True,block=self.block_index,hidden_state_index=self.block_index+1,cache_extraction_checks=cache_checks,
            zero_hooks=checks,batch1_vs4_max_probability_error=error,last_token_only=True,true_nohook=True))
        self.persist('correctness_passed');self.progress('correctness_passed')

    def checkpoint_rows(self,c):
        path=OUT/'rows'/self.key/f'{c}_{self.partition}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        done=set()
        if path.exists():
            good=0
            with path.open('rb') as f:
                for line in f:
                    try:obj=json.loads(line)
                    except json.JSONDecodeError:break
                    assert obj['config_sha256']==sha(OUT/'experiment_config.json')
                    for r in obj['rows']:
                        k=(r['index'],r['method']);assert k not in done;done.add(k)
                    good+=len(line)
            # Only discard an incomplete final write in this new run's checkpoint.
            if good<path.stat().st_size:
                with path.open('r+b') as f:f.truncate(good)
        return path,done

    def batches(self,tasks):
        batch=[]
        for t in sorted(tasks,key=lambda t:len(t['ids'])):
            limit=1 if len(t['ids'])>1024 else 2 if len(t['ids'])>512 else self.bs
            if batch and len(batch)>=limit:yield batch;batch=[]
            batch.append(t)
        if batch:yield batch

    def evaluate_partition(self,c,partition):
        assert partition==self.partition
        tasks=self.subset(c,partition);save(self.out/f'prompts_{c}_{partition}.json',tasks)
        path,done=self.checkpoint_rows(c);methods=['No-hook','Linear']+[f'lambda_{w}' for w in WEIGHTS]
        with path.open('a') as f:
            for batch in self.batches(tasks):
                self.reference={}
                for method in methods:
                    pending=[t for t in batch if (t['index'],method) not in done]
                    if not pending:continue
                    # Resume may change batch composition; geometry identity is checked within each fresh batch.
                    if len(pending)!=len(batch):self.reference={}
                    rows=self.condition(pending,method,0 if method=='No-hook' else 1)
                    for r in rows:
                        r.update(execution_provenance=f'new GPU run {self.out.name}',model=self.key)
                        p=np.array([r['candidate_logits'][l] for l in r['option_labels']]);p=np.exp(p-p.max());p/=p.sum()
                        assert abs(p[r['option_labels'].index(r['target'])]-r['normalized_target_probability'])<1e-12
                    f.write(json.dumps(dict(config_sha256=sha(OUT/'experiment_config.json'),rows=rows))+'\n');f.flush();os.fsync(f.fileno())
                    self.count+=len(rows);done.update((r['index'],method) for r in rows)
                self.progress('scoring',concept=c,partition=partition,completed_records=len(done),expected_records=len(tasks)*len(methods))
        assert len(done)==len(tasks)*len(methods)
        self.completed.append(dict(concept=c,partition=partition,questions=len(tasks)));self.persist('running')

    def run(self):
        self.load();self.restore();self.checks()
        for c in self.concepts:
            for cb in self.cbs[c].values():cb.to('cuda')
            self.evaluate_partition(c,self.partition)
            for cb in self.cbs[c].values():cb.to('cpu')
        verify();self.persist('complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen3','llama3b'],required=True);p.add_argument('--partition',choices=['validation','fit'],required=True);args=p.parse_args()
    verify();assert load(OUT/'preparation_complete.json')['passed']
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ex=Experiment(args.model,args.partition)
    def stop(sig,frame):ex.persist('interrupted_checkpoint_saved');raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise

if __name__=='__main__':main()
