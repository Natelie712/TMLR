"""Diagnostic geometry and a frozen exploratory official-test evaluation."""
import argparse,os,time,traceback,signal
import numpy as np,torch
from support import *

class Experiment(old_run.Experiment):
    def __init__(self,key):
        old=prev.stage.RESULT;prev.stage.RESULT=OUT
        try:prev.stage.Experiment.__init__(self,key,f'diagnostic-test-{time.time_ns()}')
        finally:prev.stage.RESULT=old
        self.partition='diagnostic_and_test';self.concepts=list(NAMES);self.variant='assistant_answer_parenthesis'
        self.cfg=prev.config();self.frozen=verify()
        self.vectors=torch.load(OLD/'fits'/key/'linear_vectors.pt',weights_only=True)['normalized']
        self.cbs={};self.steps={};self.completed=[];self.count=0;self.bs=4;self.reference={}
        self.manifest.update(config_sha256=sha(OUT/'frozen_config.json'),exploratory_fixed_benchmark=True,selection_bias_acknowledged=True)
        save(OUT/f'location_{key}.json',dict(path=str(self.out)));self.persist('initialized')

    def persist(self,status):
        self.manifest.update(status=status,completed=self.completed,rows=self.count,elapsed_seconds=time.time()-self.start)
        save(self.out/'manifest.json',self.manifest);save(OUT/f'status_{self.key}.json',dict(status=status,path=str(self.out),completed=self.completed,rows=self.count))

    def subset(self,c,partition):
        if partition!='benchmark':return old_run.Experiment.subset(self,c,partition)
        source=prev.data(c,'test')
        return [dict(prev.task(self.tok,self.key,c,r['index'],source[r['index']],self.variant,'test'),group_id=r['group_id'],partition='benchmark') for r in prev.SPLIT if r['concept']==c and r['partition']=='benchmark']

    def diagnostic(self,c):
        weights=[0.,.005,.01,.05,1.];tasks=self.subset(c,'validation')
        if self.key=='llama3b':
            by={}
            for t in tasks:by.setdefault(t['group_id'],t)
            tasks=[by[g] for g in sorted(by)[:16]]
        save(self.out/f'diagnostic_prompts_{c}.json',tasks)
        oldrows=[]
        for line in (OLD/'rows'/self.key/f'{c}_validation.jsonl').read_text().splitlines():oldrows+=json.loads(line)['rows']
        reference={(r['index'],r['method']):r for r in oldrows}
        for w in weights:self.cbs[c][w].to('cuda')
        rows=[];geometry=[];tensors=[];parity=[]
        for batch in self.batches(tasks):
            capture=[]
            def observe(module,args,out):capture.append((out[0] if isinstance(out,tuple) else out)[:,-1,:].clone())
            handle=self.block.register_forward_hook(observe)
            try:base,_=self.evaluate(batch,0)
            finally:handle.remove()
            assert len(capture)==1;h=capture[0];hd=h.double();step=self.steps[c]['1'];updates={}
            for w in weights:
                cb=self.cbs[c][w];updates[f'lambda_{w}']=cb.intervention(h,step).double()-hd
            updates['Linear']=(h+self.vectors[c].to(h)).double()-hd
            zero=updates['lambda_0.0'];small=updates['lambda_0.01'];n0=zero.norm(dim=-1,keepdim=True);ns=small.norm(dim=-1,keepdim=True)
            assert n0.min()>0 and ns.min()>0
            if self.key=='qwen3':
                updates['direction0_magnitude001']=zero/n0*ns
                updates['direction001_magnitude0']=small/ns*n0
            for j,(r,t) in enumerate(zip(base,batch)):
                r.update(method='No-hook',partition='validation',group_id=t['group_id'],diagnostic_only=True);rows.append(r)
                parity.append(dict(concept=c,index=t['index'],method='No-hook',probability_error=abs(r['normalized_target_probability']-reference[t['index'],'No-hook']['normalized_target_probability'])))
            for method,delta in updates.items():
                called=[]
                def inject(module,args,kwargs,out):
                    original_h=out[0] if isinstance(out,tuple) else out;before=original_h[:,-1,:]
                    assert torch.equal(before,h);assert kwargs['position_ids'][:,-1].cpu().tolist()==[len(t['ids'])-1 for t in batch]
                    changed=original_h.clone();changed[:,-1,:]=(before.double()+delta).to(before.dtype)
                    called.append((changed[:,-1,:].double()-before.double()).clone());assert torch.equal(changed[:,:-1,:],original_h[:,:-1,:])
                    return (changed,)+out[1:] if isinstance(out,tuple) else changed
                handle=self.block.register_forward_hook(inject,with_kwargs=True)
                try:rr,_=self.evaluate(batch,0)
                finally:handle.remove()
                assert len(called)==1;actual=called[0];assert torch.isfinite(actual).all()
                if method.startswith('lambda_'):
                    cb=self.cbs[c][float(method.split('_')[1])];z=cb.project(h);error=(cb.inverse(z)+cb.center-hd).norm(dim=-1)
                else:error=None
                for j,(r,t) in enumerate(zip(rr,batch)):
                    norm=float(actual[j].norm());cos0=float(torch.nn.functional.cosine_similarity(actual[j],zero[j],dim=0));cosL=float(torch.nn.functional.cosine_similarity(actual[j],updates['Linear'][j],dim=0))
                    g=dict(model=self.key,concept=c,index=t['index'],group_id=t['group_id'],method=method,displacement=norm,activation_norm=float(hd[j].norm()),norm_ratio_to_lambda0=norm/float(n0[j]),cosine_to_lambda0=cos0,cosine_to_linear=cosL,relative_update_difference_to_lambda0=float((actual[j]-zero[j]).norm()/n0[j]),reconstruction_l2=None if error is None else float(error[j]),relative_reconstruction=None if error is None else float(error[j]/hd[j].norm()))
                    geometry.append(g);r.update(method=method,partition='validation',group_id=t['group_id'],diagnostic_only=True,geometry=g);rows.append(r)
                    if method in ['Linear']+[f'lambda_{w}' for w in weights]:parity.append(dict(concept=c,index=t['index'],method=method,probability_error=abs(r['normalized_target_probability']-reference[t['index'],method]['normalized_target_probability'])))
            tensors.append(dict(indices=[t['index'] for t in batch],h=h.cpu(),updates={k:v.cpu() for k,v in updates.items()}))
        assert max(p['probability_error'] for p in parity)<1e-4,max(p['probability_error'] for p in parity)
        save(self.out/f'diagnostic_rows_{c}.json',rows);save(self.out/f'geometry_{c}.json',geometry);save(self.out/f'replay_parity_{c}.json',parity)
        torch.save(tensors,self.out/f'diagnostic_activations_updates_{c}.pt')
        for w in weights:self.cbs[c][w].to('cpu')
        self.progress('diagnostic_complete',concept=c,questions=len(tasks),replay_max_probability_error=max(p['probability_error'] for p in parity))

    def benchmark(self,c):
        selected=next(s['selected_weight'] for s in self.frozen['selected'] if s['model']==self.key and s['concept']==c)
        weights=sorted({0.,1.,selected});methods=['No-hook','Linear']+[f'lambda_{w}' for w in weights]
        for w in weights:self.cbs[c][w].to('cuda')
        tasks=self.subset(c,'benchmark');assert len(tasks)==50
        save(self.out/f'test_prompts_{c}.json',tasks)
        path=self.out/f'test_rows_{c}.jsonl';assert not path.exists()
        with path.open('w') as f:
            for batch in self.batches(tasks):
                self.reference={};rows=[]
                for method in methods:
                    rr=self.condition(batch,method,0 if method=='No-hook' else 1)
                    for r in rr:r.update(exploratory_fixed_benchmark=True,selected_weight=selected,execution_provenance=f'new frozen official-test GPU run {self.out.name}')
                    rows+=rr
                f.write(json.dumps(dict(config_sha256=sha(OUT/'frozen_config.json'),rows=rows))+'\n');f.flush();os.fsync(f.fileno());self.count+=len(rows)
        for w in weights:self.cbs[c][w].to('cpu')
        self.completed.append(dict(concept=c,questions=50,methods=methods,selected_weight=selected));self.persist('running');self.progress('test_concept_complete',concept=c)

    def run(self):
        self.load();self.restore();self.checks()
        if self.key=='qwen3':self.diagnostic('hallucination')
        else:
            for c in self.concepts:self.diagnostic(c)
        # All test choices were serialized before diagnostics or test inference.
        verify()
        for c in self.concepts:self.benchmark(c)
        verify();self.persist('complete')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen3','llama3b'],required=True);args=parser.parse_args()
    verify();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status_file=OUT/f'status_{args.model}.json'
    if status_file.exists() and load(status_file)['status']=='complete':
        print(json.dumps(dict(model=args.model,status='already_complete_no_repeat')),flush=True);return
    ex=Experiment(args.model)
    def stop(sig,frame):ex.persist('interrupted_partial_batches_saved');raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise

if __name__=='__main__':main()
