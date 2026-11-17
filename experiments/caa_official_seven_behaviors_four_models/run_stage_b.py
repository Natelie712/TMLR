"""Staged scoring checks and Linear baselines. No automatic multi-method sweep."""
import argparse,importlib.metadata,json,os,time,traceback,shutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from runtime_b import *

CONFIG_PATH=HERE/'stage_c_config.json'
CONFIG=json.loads(CONFIG_PATH.read_text())
RESULT=ROOT/'results/caa_official_seven_behaviors_four_models/staged_execution_20261030'


class Experiment:
    def __init__(self,key,mode='score-only'):
        self.key=key;self.block_index=MODELS[key]['block'];self.variant=None
        self.mode=mode
        self.out=RESULT/f'{key}_{mode}_{os.environ.get("SLURM_JOB_ID","local")}'
        self.out.mkdir(parents=True,exist_ok=False);self.start=time.time();self.bs=CONFIG['runtime']['batch_size']
        (self.out/'sources').mkdir()
        for p in [Path(__file__),HERE/'runtime_b.py',HERE/'scoring.py',CONFIG_PATH]:shutil.copy2(p,self.out/'sources'/p.name)
        self.manifest=dict(status='running',model=MODELS[key],job_id=os.getenv('SLURM_JOB_ID'),
            stage=mode,formal_sweep=False,
            config_sha256=tc.sha(CONFIG_PATH),mode=mode,
            source_sha256={p.name:tc.sha(p) for p in [Path(__file__),HERE/'runtime_b.py',HERE/'scoring.py']},
            software={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers']})
        tc.save(self.out/'manifest.json',self.manifest)
    def progress(self,stage,**kw):
        info=dict(model=self.key,stage=stage,elapsed_seconds=time.time()-self.start,**kw)
        tc.save(self.out/'progress.json',info);print(json.dumps(info),flush=True)
    def load(self):
        regressions=[p for p in RESULT.parent.glob('regression_*/manifest.json') if json.loads(p.read_text()).get('status')=='passed']
        assert regressions,'A fresh successful released-vector regression is required before this GPU validation/pilot'
        self.manifest['regression_manifest']=str(sorted(regressions)[-1])
        assert torch.cuda.is_available(),'CUDA initialization failed before model loading'
        self.manifest.update(gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count(),snapshot=tc.modelpath(self.key))
        tc.save(self.out/'manifest.json',self.manifest)
        self.tok=tc.legacy.tokenizer(self.key,tc.modelpath(self.key))
        model=AutoModelForCausalLM.from_pretrained(tc.modelpath(self.key),local_files_only=True,token=False,dtype=torch.float32,attn_implementation='eager')
        self.model=tc.place_model(model) if self.key=='llama2' else model.cuda().eval()
        self.block=self.model.model.layers[self.block_index]
        self.manifest.update(decoder_blocks=len(self.model.model.layers),intervention_class=type(self.block).__name__,hidden_state_index=self.block_index+1)
        self.progress('loaded')
    def tasks(self,split,variant=None,limit=None):
        return [task(self.tok,self.key,c,i,r,variant or self.variant,split) for c in CONCEPTS for i,r in enumerate(data(c,split)) if limit is None or i<limit]
    def forward(self,tasks,hidden=False):
        ids,mask,pos,pad=tc.tensors(self.tok,tasks)
        ret=self.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=hidden,logits_to_keep=1)
        return ret,pad
    def evaluate(self,tasks,alpha=0,vectors=None):
        rows=[];full={}
        # Batch within concept so each intervention uses exactly its own vector.
        for c in CONCEPTS:
            subset=[t for t in tasks if t['concept']==c]
            for batch in tc.batches(subset,self.bs):
                observations=[];handle=None
                if alpha:
                    bound=torch.tensor([[t['decision']['boundary']] for t in batch],device='cuda')
                    handle=self.block.register_forward_hook(tc.scoped_observer(vectors[c].cuda(),alpha,bound,observations),with_kwargs=True)
                try:ret,_=self.forward(batch)
                finally:
                    if handle:handle.remove()
                logits=ret.logits.cpu()
                if alpha:assert len(observations)==len(batch),'Intervention not triggered once per sample'
                for j,t in enumerate(batch):
                    d=t['decision'];s=score(logits[j],d['option_ids'],t['target']);top=int(logits[j].argmax())
                    spaced={k:v['spaced_ids'] for k,v in d['tokenization'].items()}
                    p=logits[j,-1].float().softmax(0)
                    top_ids=logits[j,-1].topk(10).indices.tolist()
                    row=dict(model=self.key,concept=c,index=t['index'],split=t['split'],prompt_variant=d['prompt_variant'],alpha=alpha,
                        intervention_triggered=bool(alpha),greedy_id=top,greedy_text=self.tok.decode([top]),greedy_is_legal_label=top in d['option_ids'].values(),
                        top_tokens=[dict(token_id=k,text=self.tok.decode([k]),logit=float(logits[j,-1,k]),probability=float(p[k])) for k in top_ids],
                        spaced_candidate_ids=spaced,spaced_probabilities={k:float(p[v[0]]) if len(v)==1 else None for k,v in spaced.items()},
                        displacement=observations[j] if alpha else dict(actual_displacement=[0.],positions=[],relative_displacement=[0.]),**s)
                    rows.append(row);full[f'{c}:{t["index"]}']=logits[j].clone()
        return rows,full
    def gate(self):
        checks=[]
        # The user requires a separate report before any template/candidate
        # adaptation. Diagnose the original format; never choose a fallback here.
        for variant in ['legacy_parenthesis']:
            tasks=self.tasks('generate',variant,8);rows,_=self.evaluate(tasks)
            mass=np.array([r['all_option_probability_mass'] for r in rows]);hit=np.mean([r['greedy_is_legal_label'] for r in rows])
            passed=hit>=.9 and np.median(mass)>=.1 and mass.min()>1e-8
            checks.append(dict(variant=variant,passed=bool(passed),legal_greedy_fraction=float(hit),median_all_option_mass=float(np.median(mass)),minimum_all_option_mass=float(mass.min())))
            tc.save(self.out/f'gate_{variant}_rows.json',rows);tc.save(self.out/f'gate_{variant}_prompts.json',tasks)
            tc.save(self.out/'prompt_gate.json',dict(selected=variant if passed else None,candidates=checks))
            if passed:self.variant=variant;break
        if not self.variant:
            self.manifest.update(status='scoring_mismatch_requires_report',automatic_template_change=False)
            tc.save(self.out/'manifest.json',self.manifest)
            raise ValueError('Original candidate/template gate did not pass; report evidence before any adaptation')
        self.manifest['selected_prompt_variant']=self.variant;tc.save(self.out/'manifest.json',self.manifest)
        generations=[]
        for t in self.tasks('generate',limit=1):
            cur=dict(t);cur['ids']=list(t['ids']);new=[]
            for _ in range(3):
                ret,_=self.forward([cur]);v=int(ret.logits[0,-1].argmax());new.append(v);cur['ids'].append(v)
            generations.append(dict(concept=t['concept'],prompt=t['decision']['rendered'],generated_ids=new,generated_text=self.tok.decode(new),first_is_legal=new[0] in t['decision']['option_ids'].values()))
        tc.save(self.out/'answer_position_continuations.json',generations)
        assert all(r['first_is_legal'] for r in generations),'Greedy continuation does not confirm label decision position'
        self.progress('prompt_gate_passed',variant=self.variant)
    def parity(self):
        tasks=self.tasks('generate',limit=1);vec=torch.randn(self.model.config.hidden_size,device='cuda');vec/=vec.norm()
        checks=[]
        for alpha in [0,-1,1]:
            scalar={}
            for t in tasks:
                h=self.block.register_forward_hook(tc.translation_hook(vec,alpha,t['decision']['boundary']),with_kwargs=True) if alpha else None
                try:ret,_=self.forward([t])
                finally:
                    if h:h.remove()
                scalar[t['concept']]=ret.logits.cpu()
            for batch in tc.batches(tasks,self.bs):
                obs=[];bound=torch.tensor([[t['decision']['boundary']] for t in batch],device='cuda')
                h=self.block.register_forward_hook(tc.scoped_observer(vec,alpha,bound,obs),with_kwargs=True) if alpha else None
                try:ret,_=self.forward(batch)
                finally:
                    if h:h.remove()
                for j,t in enumerate(batch):
                    x=ret.logits[j:j+1].cpu();ref=scalar[t['concept']];error=float((x-ref).abs().max())
                    pe=abs(score(x,t['decision']['option_ids'],t['target'])['normalized_target_probability']-score(ref,t['decision']['option_ids'],t['target'])['normalized_target_probability'])
                    checks.append(dict(concept=t['concept'],alpha=alpha,max_full_vocab_logit_error=error,normalized_probability_error=pe))
                    assert error<.002 and pe<.0001
        tc.save(self.out/'batch_hook_parity.json',checks);self.progress('batch_and_hook_parity_passed')
    def nohook_test_diagnostics(self):
        variants=list(dict.fromkeys([self.variant,'legacy_parenthesis']))
        for variant in variants:
            tasks=self.tasks('test',variant);rows,logits=self.evaluate(tasks)
            binary=[r for r in rows if r['option_labels']==['A','B']]
            assert len(binary)==342 and max(abs(r['B_minus_A_stable']) for r in binary)<1e-14
            assert all(r['scheme_A_original_probability'] is not None for r in rows)
            tc.save(self.out/f'nohook_test_{variant}.json',rows)
            tc.save(self.out/f'nohook_test_{variant}_prompts.json',tasks)
            torch.save(logits,self.out/f'nohook_test_{variant}_full_logits.pt')
        self.progress('all350_nohook_scorer_diagnostics_complete',steered_test_rows=0)
    def extract(self):
        self.raw={};checks=[];stability=[]
        with (self.out/'training_prompts.jsonl').open('w') as prompts:
            for c in CONCEPTS:
                pairs=data(c,'generate');tasks=[]
                for i,r in enumerate(pairs):
                    t=task(self.tok,self.key,c,i,r,self.variant,'generate')
                    for sign,label in [(1,t['target']),(-1,t['nonmatching'])]:
                        z=dict(t,sign=sign,train_row=len(tasks),answer_label=label,ids=t['decision']['completion_ids'][label]);tasks.append(z)
                        prompts.write(json.dumps(dict(concept=c,index=i,sign=sign,answer_label=label,input_ids=z['ids'],extraction_position=z['decision']['extraction_position'],rendered=self.tok.decode(z['ids'])),ensure_ascii=True)+'\n')
                acts=torch.empty(len(tasks),self.model.config.hidden_size,dtype=torch.float32)
                for bi,batch in enumerate(tc.batches(tasks,self.bs)):
                    _,_,_,pads=tc.tensors(self.tok,batch)
                    indices=torch.tensor([t['decision']['extraction_position']+p for t,p in zip(batch,pads)],device='cuda');capture=[]
                    def hook(module,args,output):
                        x=output[0] if isinstance(output,tuple) else output
                        capture.append(x[torch.arange(len(batch),device=x.device),indices.to(x.device)].clone())
                    h=self.block.register_forward_hook(hook)
                    try:ret,_=self.forward(batch,hidden=bi==0)
                    finally:h.remove()
                    assert len(capture)==1
                    if bi==0:
                        hs=ret.hidden_states[self.block_index+1];expected=hs[torch.arange(len(batch),device=hs.device),indices.to(hs.device)]
                        assert torch.equal(capture[0],expected)
                        checks.append(dict(concept=c,block=self.block_index,hidden_state_index=self.block_index+1,max_error=0))
                    for j,t in enumerate(batch):acts[t['train_row']]=capture[0][j].cpu()
                    if bi%80==0:self.progress('extracting',concept=c,batch=bi,total_rows=len(tasks))
                delta=acts[0::2]-acts[1::2];v=delta.mean(0);assert v.norm()>1e-8 and torch.isfinite(v).all();self.raw[c]=v
                torch.save(dict(positive=acts[0::2],negative=acts[1::2],source_indices=list(range(len(pairs)))),self.out/f'train_activations_{c}.pt')
                cos=lambda a,b:float(torch.nn.functional.cosine_similarity(a.double(),b.double(),dim=0))
                stability.append(dict(concept=c,pairs=len(pairs),raw_norm=float(v.norm()),cosine_first12_vs_full=cos(delta[:12].mean(0),v),cosine_disjoint_halves=cos(delta[::2].mean(0),delta[1::2].mean(0))))
                self.progress('concept_extracted',concept=c,pairs=len(pairs))
        norm=torch.stack([self.raw[c].norm() for c in CONCEPTS]).mean()
        self.vectors={c:v*norm/v.norm() for c,v in self.raw.items()}
        tc.save(self.out/'extraction_index_checks.json',checks);tc.save(self.out/'direction_stability.json',stability)
        tc.save(self.out/'calibration.json',[dict(concept=c,raw_norm=float(self.raw[c].norm()),calibrated_norm=float(v.norm()),mean_seven_raw_norm=float(norm)) for c,v in self.vectors.items()])
        torch.save(dict(raw=self.raw,normalized=self.vectors,training_source='official generate; all source rows'),self.out/'newly_extracted_vectors.pt')
        if self.key=='llama2':
            released=torch.load(ROOT/'experiments/caa_reproduction/reference/normalized_vectors/corrigible-neutral-HHH/vec_layer_13_Llama-2-7b-chat-hf.pt',weights_only=True)
            own=self.vectors['corrigible-neutral-HHH'];tc.save(self.out/'released_vs_own_corrigibility.json',dict(released_norm=float(released.norm()),own_norm=float(own.norm()),cosine=float(torch.nn.functional.cosine_similarity(released.double(),own.double(),dim=0)),released_used_for_training_or_pilot=False))
        self.progress('seven_vectors_extracted')
    def pilot(self):
        tasks=self.tasks('generate',limit=8);rows=[]
        tc.save(self.out/'pilot_prompts.json',tasks)
        for alpha in [0,-1,1]:
            result,logits=self.evaluate(tasks,alpha,self.vectors);rows.extend(result)
            torch.save(logits,self.out/f'pilot_logits_alpha_{alpha}.pt')
        tc.save(self.out/'linear_pilot_rows.json',rows)
        summary=[]
        for c in CONCEPTS:
            means={a:float(np.mean([r['normalized_target_probability'] for r in rows if r['concept']==c and r['alpha']==a])) for a in [-1,0,1]}
            summary.append(dict(model=self.key,concept=c,questions=8,classification='in-sample pilot, not held-out performance',negative=means[-1],no_hook=means[0],positive=means[1],delta_positive_pp=100*(means[1]-means[0]),delta_negative_pp=100*(means[-1]-means[0])))
        tc.save(self.out/'linear_pilot_summary.json',summary)
        self.manifest.update(status='passed',elapsed_seconds=time.time()-self.start,selected_prompt_variant=self.variant,train_questions=sum(len(data(c,'generate')) for c in CONCEPTS),pilot_rows=len(rows),steered_official_test_rows=0)
        tc.save(self.out/'manifest.json',self.manifest);self.progress('linear_pilot_complete')

    def linear_baseline(self):
        """Authorized Step 4: Llama-2 only, original 350 official test questions."""
        assert self.key=='llama2','Expand other-model baselines only after reviewing Llama-2 reliability'
        tasks=self.tasks('test');rows=[]
        tc.save(self.out/'linear_test_prompts.json',tasks)
        for alpha in [0,-1,1]:
            result,logits=self.evaluate(tasks,alpha,self.vectors);rows.extend(result)
            torch.save(logits,self.out/f'linear_test_logits_alpha_{alpha}.pt')
            tc.save(self.out/'linear_test_rows.json',rows)
        comparisons=[];summary=[];rng=np.random.default_rng(42)
        for c in CONCEPTS:
            byalpha={a:sorted([r for r in rows if r['concept']==c and r['alpha']==a],key=lambda r:r['index']) for a in [-1,0,1]}
            values={a:np.array([r['normalized_target_probability'] for r in byalpha[a]]) for a in byalpha}
            assert all(len(v)==50 for v in values.values())
            boot=rng.integers(0,50,(10000,50));pos=(values[1]-values[0])*100;neg=(values[-1]-values[0])*100
            refpath=ROOT/f'experiments/caa_reproduction/reference/vectors/{c}/vec_layer_13_Llama-2-7b-chat-hf.pt'
            ref=torch.load(refpath,weights_only=True)
            comparisons.append(dict(concept=c,released_raw_sha256=tc.sha(refpath),released_raw_norm=float(ref.norm()),
                extracted_raw_norm=float(self.raw[c].norm()),raw_cosine=float(torch.nn.functional.cosine_similarity(ref.double(),self.raw[c].double(),dim=0)),
                calibrated_new_norm=float(self.vectors[c].norm()),released_substituted_for_new=False))
            summary.append(dict(model=self.key,concept=c,questions=50,metric='Scheme B all-option normalized target-label probability',
                negative=float(values[-1].mean()),no_hook=float(values[0].mean()),positive=float(values[1].mean()),
                delta_positive_pp=float(pos.mean()),delta_negative_pp=float(neg.mean()),
                delta_positive_ci95_pp=np.quantile(pos[boot].mean(1),[.025,.975]).tolist(),
                delta_negative_ci95_pp=np.quantile(neg[boot].mean(1),[.025,.975]).tolist(),
                mean_positive_displacement=float(np.mean([d for r in byalpha[1] for d in r['displacement']['actual_displacement']])),
                mean_negative_displacement=float(np.mean([d for r in byalpha[-1] for d in r['displacement']['actual_displacement']]))))
        tc.save(self.out/'linear_test_summary.json',summary);tc.save(self.out/'released_vs_extracted_all7.json',comparisons)
        # A paired fresh control on the same 50 Corrigibility prompts distinguishes
        # extraction differences from a change in the running evaluator.
        released=torch.load(ROOT/'experiments/caa_reproduction/reference/normalized_vectors/corrigible-neutral-HHH/vec_layer_13_Llama-2-7b-chat-hf.pt',weights_only=True)
        corr=[t for t in tasks if t['concept']=='corrigible-neutral-HHH'];ref_rows=[]
        for alpha in [-1,1]:
            rr,_=self.evaluate(corr,alpha,{'corrigible-neutral-HHH':released});ref_rows.extend(rr)
        tc.save(self.out/'released_corrigibility_fresh_control.json',ref_rows)
        own=next(r for r in summary if r['concept']=='corrigible-neutral-HHH')
        deltas={str(a):float(np.mean([r['normalized_target_probability'] for r in ref_rows if r['alpha']==a]))-own['negative' if a==-1 else 'positive'] for a in [-1,1]}
        comparison=next(r for r in comparisons if r['concept']=='corrigible-neutral-HHH')
        needs_diagnosis=max(abs(v) for v in deltas.values())>.005 or comparison['raw_cosine']<.99
        tc.save(self.out/'corrigibility_reliability_gate.json',dict(released_minus_new_probability=deltas,raw_cosine=comparison['raw_cosine'],
            needs_root_cause_analysis=needs_diagnosis,thresholds=dict(probability_absolute=.005,raw_cosine_min=.99),released_vector_replacement=False,
            expansion_allowed=False,reason='Review all seven Linear responses and four-model scoring gates before multi-method expansion'))
        self.manifest.update(status='linear_baselines_complete_requires_review',elapsed_seconds=time.time()-self.start,
            selected_prompt_variant=self.variant,train_questions=4911,steered_official_test_rows=700,
            own_vector_test_rows=len(rows),released_control_extra_rows=len(ref_rows),multi_method_sweep_started=False)
        tc.save(self.out/'manifest.json',self.manifest);self.progress('llama2_seven_linear_baselines_complete',needs_corrigibility_diagnosis=needs_diagnosis)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=list(MODELS),required=True)
    parser.add_argument('--mode',choices=['score-only','pilot','linear-baseline'],default='score-only');args=parser.parse_args()
    torch.manual_seed(42);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ex=Experiment(args.model,args.mode)
    try:
        with torch.inference_mode():
            ex.load();ex.gate();ex.parity();ex.nohook_test_diagnostics()
            if args.mode=='score-only':
                ex.manifest.update(status='scoring_passed',selected_prompt_variant=ex.variant,elapsed_seconds=time.time()-ex.start)
                tc.save(ex.out/'manifest.json',ex.manifest);ex.progress('scoring_checks_complete')
            else:
                ex.extract()
                if args.mode=='pilot':ex.pilot()
                else:ex.linear_baseline()
    except Exception:
        if ex.manifest['status']!='scoring_mismatch_requires_report':ex.manifest['status']='failed'
        ex.manifest['error']=traceback.format_exc();tc.save(ex.out/'manifest.json',ex.manifest);raise


if __name__=='__main__':main()
