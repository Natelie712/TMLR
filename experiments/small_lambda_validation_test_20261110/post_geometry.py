"""Post-run arithmetic diagnostics on saved activations; no outcome selection."""
import joblib,numpy as np,torch
from support import *
from analyze import write

def main():
    verify();torch.set_num_threads(2);decomposition=[];casting=[]
    for key in ['qwen3','llama3b']:
        loc=OUT/f'location_{key}.json'
        if not loc.exists():continue
        run=Path(load(loc)['path'])
        for file in run.glob('diagnostic_activations_updates_*.pt'):
            c=file.stem.removeprefix('diagnostic_activations_updates_');batches=torch.load(file,weights_only=True)
            h=torch.cat([b['h'] for b in batches]);ids=[i for b in batches for i in b['indices']];hd=h.double()
            for method in batches[0]['updates']:
                requested=torch.cat([b['updates'][method] for b in batches]);actual=(hd+requested).to(h.dtype).double()-hd
                err=(actual-requested).norm(dim=-1)
                for i,index in enumerate(ids):casting.append(dict(model=key,concept=c,index=index,method=method,absolute_cast_error=float(err[i]),relative_cast_error=float(err[i]/requested[i].norm())))
            for w in [0.,.005,.01,.05,1.]:
                cb=prev.AttenuatedCurveball.restore(joblib.load(prev.fit_path(key,c,w)));z=cb.project(h);step=load(OLD/'fits'/key/'positive_steps.json')['steps'][c]['1']
                # Exact weighted polynomial inverse difference: constant terms cancel.
                a=.01*(z@cb.train_z.T);b=.01*step*(cb.direction@cb.train_z.T)
                linear=(2*b@cb.dual).expand(len(h),-1)
                nonlinear=(w*(2*a*b+b*b))@cb.dual
                full=cb.inverse(z+step*cb.direction)-cb.inverse(z)
                rel=(full-linear-nonlinear).norm(dim=-1)/full.norm(dim=-1)
                assert float(rel.max())<1e-6,(key,c,w,float(rel.max()))
                for i,index in enumerate(ids):
                    decomposition.append(dict(model=key,concept=c,index=index,weight=w,inverse_linear_component_norm=float(linear[i].norm()),inverse_quadratic_component_norm=float(nonlinear[i].norm()),inverse_quadratic_to_linear_update_ratio=float(nonlinear[i].norm()/linear[i].norm()),total_unrounded_update_norm=float(full[i].norm()),decomposition_relative_error=float(rel[i])))
    write('inverse_update_components_per_question.csv',decomposition);write('cast_error_per_question.csv',casting)
    summaries=[]
    for key in ['qwen3','llama3b']:
        for c in NAMES:
            for w in [0.,.005,.01,.05,1.]:
                rr=[r for r in decomposition if r['model']==key and r['concept']==c and r['weight']==w]
                if rr:
                    a=np.array([r['inverse_quadratic_to_linear_update_ratio'] for r in rr]);summaries.append(dict(model=key,concept=c,weight=w,questions=len(rr),mean_quadratic_to_linear_update_ratio=float(a.mean()),median_ratio=float(np.median(a)),maximum_ratio=float(a.max()),max_decomposition_relative_error=max(r['decomposition_relative_error'] for r in rr)))
    write('inverse_update_component_summary.csv',summaries)
    save(OUT/'post_geometry_status.json',dict(questions_with_component_records=len(decomposition),max_cast_relative_error=max((r['relative_cast_error'] for r in casting),default=None),passed=True))
    print(json.dumps(dict(component_summary=summaries,max_cast_relative_error=max((r['relative_cast_error'] for r in casting),default=None))),flush=True)

if __name__=='__main__':main()
