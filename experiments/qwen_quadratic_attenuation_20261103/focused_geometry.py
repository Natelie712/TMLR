"""Focused, validation-only geometry controls. Never edits the CAA baseline."""
import importlib.util, json, os, shutil, signal, sys, time, traceback
from pathlib import Path
import joblib
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('attenuation_prior_runner',Path(__file__).with_name('run.py'))
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from weighted import AttenuatedCurveball

ROOT = prior.ROOT
OUT = prior.OUT / 'focused_diagnostics_20261103'
CFG = prior.load(OUT / 'config.json')
C = 'corrigible-neutral-HHH'
SOURCE = Path(CFG['fit_source'])
save = prior.save


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)


class Focus(prior.stage.Experiment):
    def __init__(self):
        old = prior.stage.RESULT
        prior.stage.RESULT = OUT
        super().__init__('qwen3', 'focused-geometry')
        prior.stage.RESULT = old
        self.variant = 'assistant_answer_parenthesis'
        self.vectors = torch.load(SOURCE / 'fit_only_linear_vectors.pt', weights_only=True)['normalized']
        self.rows = []; self.geometry = {}; self.reference = {}; self.identity = []
        self.v = self.vectors[C].double()
        self.budget = float(self.v.norm())
        for path in [Path(__file__), OUT / 'config.json', OUT / 'protocol.md']:
            shutil.copy2(path, self.out / 'sources' / path.name)

    def persist(self, status):
        save(self.out / 'rows.json', self.rows)
        self.manifest.update(status=status, elapsed_seconds=time.time()-self.start,
                             completed_rows=len(self.rows), validation_only=True)
        save(self.out / 'manifest.json', self.manifest)
        save(OUT / 'run_location.json', dict(path=str(self.out), status=status))

    def evaluate_control(self, tasks, method, sign, cb=None, mode='original', vector=None, step=None):
        """Use the frozen scorer; only the explicitly labelled update is varied."""
        result=[]
        for batch in prior.stage.tc.batches(tasks, self.bs):
            observations=[]; calls=[]
            def hook(module, args, kwargs, out):
                calls.append(1)
                h=out[0] if isinstance(out, tuple) else out
                positions=kwargs['position_ids']
                bounds=torch.tensor([[t['decision']['boundary']] for t in batch], device=h.device)
                mask=positions>=bounds
                before=h[mask].clone(); hd=before.double()
                if mode=='nohook': after=before
                elif cb is not None:
                    candidate=cb.intervention(before, step)
                    raw_delta=candidate.double()-hd
                    assert raw_delta.norm(dim=-1).min()>1e-10
                    if mode=='matched':
                        after=(hd+raw_delta*self.budget/raw_delta.norm(dim=-1,keepdim=True)).to(before.dtype)
                    else: after=candidate
                else: after=(hd+vector.to(hd)).to(before.dtype)
                delta=after.double()-hd
                h[mask]=after
                assert torch.isfinite(after).all()
                cursor=0
                for j,t in enumerate(batch):
                    ps=positions[j,mask[j]].cpu().tolist(); n=len(ps); sl=slice(cursor,cursor+n);cursor+=n
                    expected=list(range(t['decision']['boundary'],len(t['ids'])))
                    assert ps==expected, (ps,expected)
                    original=hd[sl].cpu()
                    if mode=='nohook': self.reference[t['index']]=dict(activation=original,positions=ps)
                    ref=self.reference[t['index']]
                    assert ref['positions']==ps
                    assert torch.equal(ref['activation'],original), 'Pre-intervention activations changed'
                    if sign:
                        linear_actual=(before[sl]+sign*self.vectors[C].to(before)).double()-hd[sl]
                        cs=cosine(delta[sl],linear_actual)
                        alignment=(delta[sl]*linear_actual).sum(-1)/linear_actual.square().sum(-1)
                    else: cs=torch.zeros(n,dtype=torch.float64,device=h.device);alignment=cs
                    observations.append(dict(positions=ps,activation_norm=hd[sl].norm(dim=-1).cpu().tolist(),
                        actual_displacement=delta[sl].norm(dim=-1).cpu().tolist(),
                        cosine_to_actual_signed_linear=cs.cpu().tolist(),signed_parallel_multiplier=alignment.cpu().tolist()))
            handle=self.block.register_forward_hook(hook,with_kwargs=True)
            try: rr,_=self.evaluate(batch,0)
            finally: handle.remove()
            assert len(calls)==1 and len(observations)==len(batch)
            for r,t,g in zip(rr,batch,observations):
                r.update(method=method,alpha=sign,condition=mode,group_id=t['group_id'],partition='validation',
                         native_parameter=step,displacement=g,intervention_triggered=bool(sign),
                         execution_provenance='new focused GPU run '+str(self.manifest['job_id']))
                result.append(r)
        return result

    def geometric_checks(self,tasks,cbs):
        ordered=[self.reference[t['index']]['activation'] for t in tasks]
        h=torch.cat(ordered).cuda().float(); lengths=[len(x) for x in ordered]
        per_token=[]; checks=[]
        for w,cb in cbs.items():
            z=cb.project(h); rec=cb.inverse(z)+cb.center
            err=(rec-h.double()).norm(dim=-1); hn=h.double().norm(dim=-1)
            direct=cb.intervention(h,0)
            explicit=(h.double()+cb.inverse(z+0*cb.direction)-cb.inverse(z)).to(h.dtype)
            # The operation ordering above intentionally checks cancellation as well as the fast path.
            zp=cb.fit.transform(h.cpu().double().numpy()-cb.mean)
            ip=cb.fit.inverse_transform(zp)
            pe=float(np.max(np.abs(z.cpu().numpy()-zp)))
            ie=float(np.max(np.abs(cb.inverse(z).cpu().numpy()-ip)))
            np.testing.assert_allclose(z.cpu(),zp,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(cb.inverse(z).cpu(),ip,rtol=1e-6,atol=1e-5)
            assert torch.equal(direct,h)
            assert float((explicit-h).abs().max())<1e-5
            cur=0
            for t,n in zip(tasks,lengths):
                for j,pos in enumerate(self.reference[t['index']]['positions']):
                    k=cur+j
                    per_token.append(dict(index=t['index'],group_id=t['group_id'],weight=w,token_position=pos,
                        activation_norm=float(hn[k]),absolute_reconstruction_error=float(err[k]),
                        relative_reconstruction_error=float(err[k]/hn[k]),projection_norm=float(z[k].norm()),
                        zero_fastpath_max_error=float((direct[k]-h[k]).abs().max()),
                        zero_explicit_max_error=float((explicit[k]-h[k]).abs().max())))
                cur+=n
            checks.append(dict(weight=w,positions=len(h),zero_fastpath_bitwise_identity=True,
                zero_explicit_max_error=float((explicit-h).abs().max()),projection_sklearn_max_error=pe,
                inverse_sklearn_max_error=ie,reconstruction_absolute_rms=float(err.square().mean().sqrt()),
                activation_rms=float(hn.square().mean().sqrt()),relative_frobenius=float(err.norm()/hn.norm())))
        save(self.out/'reconstruction_per_token.json',per_token)
        save(self.out/'identity_and_parity_checks.json',checks)
        torch.save(dict(tasks=[dict(index=t['index'],group_id=t['group_id']) for t in tasks],
                        reference=self.reference),self.out/'validation_intervention_activations.pt')

    def lambda_zero_analysis(self,cb):
        beta=.02
        x=cb.fit.X_fit_; v=self.v.numpy()
        # For k0=1+beta*x'y, these are the retained orthonormal PCA axes.
        axes=np.sqrt(beta)*x.T@(cb.fit.eigenvectors_/np.sqrt(cb.fit.eigenvalues_))
        np.testing.assert_allclose(axes.T@axes,np.eye(axes.shape[1]),atol=1e-8)
        projection=(v@axes)@axes.T
        ideal=cb.direction.cpu().numpy()@axes.T/np.sqrt(beta)
        shrink=beta*cb.fit.eigenvalues_/(beta*cb.fit.eigenvalues_+1.)
        predicted=(cb.direction.cpu().numpy()*shrink)@axes.T/np.sqrt(beta)
        analytic=(beta*cb.direction@cb.train_z.T@cb.dual).cpu().numpy()
        np.testing.assert_allclose(analytic,predicted,atol=1e-7,rtol=1e-6)
        cos=lambda a,b:float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
        self.projected=torch.tensor(projection/np.linalg.norm(projection)*self.budget,device='cuda')
        self.analytic=torch.tensor(analytic,device='cuda')
        h=torch.cat([r['activation'] for r in self.reference.values()]).cuda().float()
        affine_checks=[]
        for a in [-1,1]:
            step=CFG['native_parameters'][str(a)]
            expected=(h.double()+step*self.analytic).float()
            observed=cb.intervention(h,step)
            error=float((expected-observed).abs().max())
            assert error<1e-4
            affine_checks.append(dict(alpha=a,max_coordinate_error=error))
        stats=dict(retained_dimensions=axes.shape[1],activation_dimensions=len(v),
            retained_direction_norm_fraction=float(np.linalg.norm(projection)/np.linalg.norm(v)),
            discarded_direction_norm_fraction=float(np.linalg.norm(v-projection)/np.linalg.norm(v)),
            retained_variance_fraction=float(cb.fit.eigenvalues_.sum()/(beta*np.square(x).sum())),
            projected_linear_cosine=cos(projection,v),analytic_update_linear_cosine=cos(analytic,v),
            inverse_vs_projected_direction_cosine=cos(analytic,projection),
            ideal_inverse_vs_projection_cosine=cos(ideal,projection),
            ridge_shrink_min=float(shrink.min()),ridge_shrink_max=float(shrink.max()),
            analytic_unit_step_norm=float(np.linalg.norm(analytic)),
            inverse_formula_max_error=float(np.max(np.abs(analytic-predicted))),
            affine_update_checks=affine_checks,
            explanation='For lambda0 the update is constant in h: step * beta * direction * Z_train^T * dual. Reconstruction offsets cancel exactly. Truncation and ridge shrinkage remain in the update.')
        save(self.out/'lambda_zero_decomposition.json',stats)

    def run(self):
        self.load()
        tasks=prior.Ablation.subset(self,C,'validation')
        assert len(tasks)==52
        save(self.out/'prompts.json',tasks)
        self.rows+=self.evaluate_control(tasks,'No-hook',0,mode='nohook')
        for a in [-1,1]:
            rr,_=self.evaluate(tasks,a,self.vectors)
            task_by_index={t['index']:t for t in tasks}
            for r in rr:
                t=task_by_index[r['index']]
                r.update(method='Linear',condition='validated_baseline',group_id=t['group_id'],partition='validation',
                         execution_provenance='fresh unchanged CAA evaluator')
                assert r['displacement']['positions']==self.reference[r['index']]['positions']
            self.rows+=rr
        cbs={w:AttenuatedCurveball.restore(joblib.load(SOURCE/f'fit_{C}_lambda_{w}.joblib'),'cuda') for w in [0.,.5,1.]}
        self.geometric_checks(tasks,cbs);self.lambda_zero_analysis(cbs[0.])
        self.persist('baseline_and_geometry_complete')
        # Independent intervention-path check using the same Linear direction and magnitude.
        for a in [-1,1]:
            rr=self.evaluate_control(tasks,'Linear_hook_control',a,mode='matched',vector=a*self.v.cuda())
            base={r['index']:r for r in self.rows if r['method']=='Linear' and r['alpha']==a}
            errors=[abs(r['normalized_target_probability']-base[r['index']]['normalized_target_probability']) for r in rr]
            assert max(errors)<1e-4
            self.rows+=rr
        for w,cb in cbs.items():
            for mode in ['original','matched']:
                for a in [-1,1]:
                    step=CFG['native_parameters'][str(a)]
                    self.rows+=self.evaluate_control(tasks,f'lambda_{w}',a,cb=cb,mode=mode,step=step)
                    self.persist('running_checkpoint_saved')
                    self.progress('condition_complete',weight=w,condition=mode,alpha=a)
        for a in [-1,1]:
            magnitude=abs(CFG['native_parameters'][str(a)])*float(self.analytic.norm())
            self.rows+=self.evaluate_control(tasks,'Linear_large_displacement',a,mode='magnitude_control',vector=a*self.v.cuda()*magnitude/self.budget)
            self.rows+=self.evaluate_control(tasks,'PCA_projected_Linear',a,mode='truncation_control',vector=a*self.projected)
            self.persist('running_checkpoint_saved')
        # Explicit zero-step hooks on the first frozen-length batch, for every lambda.
        for w,cb in cbs.items():
            # One batch by construction; use the same batch shape as cached no-hook.
            first=next(prior.stage.tc.batches(tasks,self.bs))
            def checked_zero(module,args,kwargs,out):
                h=out[0] if isinstance(out,tuple) else out
                bounds=torch.tensor([[t['decision']['boundary']] for t in first],device=h.device)
                mask=kwargs['position_ids']>=bounds;before=h[mask].clone()
                h[mask]=cb.intervention(before,0);assert torch.equal(h[mask],before)
                return out
            handle=self.block.register_forward_hook(checked_zero,with_kwargs=True)
            try: rr,_=self.evaluate(first,0)
            finally: handle.remove()
            base={r['index']:r for r in self.rows if r['method']=='No-hook'}
            error=max(abs(r['normalized_target_probability']-base[r['index']]['normalized_target_probability']) for r in rr)
            assert error<1e-7
            self.identity.append(dict(weight=w,questions=len(first),max_probability_error=error))
        save(self.out/'zero_hook_model_checks.json',self.identity)
        old=prior.load(Path(CFG['previous_run'])/'validation_rows.json')
        lookup={(r['method'],r['alpha'],r['index']):r for r in old}
        replay=[]
        for r in self.rows:
            if r['condition'] in ['original','nohook','validated_baseline']:
                ref=lookup[r['method'],r['alpha'],r['index']]
                replay.append(abs(r['normalized_target_probability']-ref['normalized_target_probability']))
        assert max(replay)<1e-4
        save(self.out/'replay_checks.json',dict(rows=len(replay),max_probability_error=max(replay)))
        self.persist('complete')


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for path,digest in CFG['frozen_inputs'].items(): assert prior.stage.tc.sha(ROOT/path)==digest,path
    ex=Focus()
    def stop(sig,frame):
        ex.persist('interrupted_checkpoint_saved');raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise


if __name__=='__main__':main()
