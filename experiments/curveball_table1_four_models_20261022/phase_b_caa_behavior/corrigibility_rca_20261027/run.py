"""Immutable-input, Llama-2-only CAA pipeline bridge and vector/data crossover."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from prepare import HERE, PHASE, ROOT, REF, SCRATCH, OUT, save, digest
sys.path[:0] = [str(PHASE),str(ROOT/'experiments/caa_reproduction')]
from common import decision, tokenizer, snapshot, score_logits, translation_hook

ORIGINAL_VECTOR = REF/'normalized_vectors/corrigible-neutral-HHH/vec_layer_13_Llama-2-7b-chat-hf.pt'

def place_model(model):
    """Keep FP32; use two 24-GiB GPUs if no high-memory GPU is available."""
    if torch.cuda.device_count()==1:
        return model.to('cuda').eval()
    assert torch.cuda.device_count()==2
    def move(value,device):
        if isinstance(value,torch.Tensor):return value.to(device)
        if isinstance(value,tuple):return tuple(move(x,device) for x in value)
        if isinstance(value,list):return [move(x,device) for x in value]
        if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
        return value
    def transfer(device):
        def hook(module,args,kwargs):return move(args,device),move(kwargs,device)
        return hook
    model.model.embed_tokens.to('cuda:0')
    if hasattr(model.model,'rotary_emb'):model.model.rotary_emb.to('cuda:0')
    for i,layer in enumerate(model.model.layers):
        device='cuda:0' if i<16 else 'cuda:1'
        layer.to(device)
        if i>=16:layer.register_forward_pre_hook(transfer(device),with_kwargs=True)
    model.model.norm.to('cuda:0')
    model.model.norm.register_forward_pre_hook(transfer('cuda:0'),with_kwargs=True)
    model.lm_head.to('cuda:0')
    assert all(p.dtype==torch.float32 for p in model.parameters())
    return model.eval()

def setup(stage):
    out=OUT/stage;out.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available()
    torch.manual_seed(42);np.random.seed(42)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=dict(status='running',stage=stage,started=time.time(),job_id=os.getenv('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count(),
        placement='single GPU' if torch.cuda.device_count()==1 else 'blocks 0-15 GPU0; 16-31 GPU1; final norm/head GPU0',
        dtype='float32',tf32=False,test_used=False,
        software={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers']},
        source_sha256=digest(Path(__file__)),config_sha256=digest(HERE/'config.json'),
        original_vector_path=str(ORIGINAL_VECTOR),original_vector_sha256=digest(ORIGINAL_VECTOR),
        model_snapshot=snapshot('llama2'))
    save(out/'manifest.json',manifest)
    return out,manifest

def finish(out,m,**kwargs):
    m.update(status='passed',elapsed_seconds=time.time()-m['started'],**kwargs)
    save(out/'manifest.json',m);print(json.dumps(m,indent=2),flush=True)

def original():
    assert transformers.__version__=='4.33.2'
    out,m=setup('original_runtime')
    sys.path.insert(0,str(REF))
    import llama_wrapper
    from prompting_with_steering import process_item_ab
    from utils.tokenize import tokenize_llama_chat
    llama_wrapper.get_model_path=lambda size,is_base:snapshot('llama2')
    if torch.cuda.device_count()==2:
        from types import SimpleNamespace
        original_loader=llama_wrapper.AutoModelForCausalLM.from_pretrained
        def sharded_loader(*args,**kwargs):
            model=place_model(original_loader(*args,**kwargs))
            # The original wrapper unconditionally calls .to('cuda'). Preserve
            # the already assigned FP32 placement only for this exact call.
            real_to=model.to
            def placed_to(*args,**kwargs):
                if args==('cuda',) and not kwargs:return model
                return real_to(*args,**kwargs)
            model.to=placed_to
            return model
        llama_wrapper.AutoModelForCausalLM=SimpleNamespace(from_pretrained=sharded_loader)
    wrapper=llama_wrapper.LlamaWrapper(None,size='7b',use_chat=True)
    wrapper.set_save_internal_decodings(False)
    tok=wrapper.tokenizer;vec=torch.load(ORIGINAL_VECTOR,weights_only=True).cuda()
    cases=[x for x in json.loads((HERE/'evaluation_manifest.json').read_text()) if x['dataset']=='original50']
    logits={};rows=[];prompts=[]
    with torch.inference_mode():
        for i,v in enumerate(cases):
            dec=decision(tok,'llama2',v['question'])
            assert dec['input_ids']==tokenize_llama_chat(tok,v['question'],'(')
            prompts.append(dict(index=i,**dec))
            for alpha in [-1,0,1]:
                wrapper.reset_all()
                if alpha:wrapper.set_add_activations(13,alpha*vec)
                captured=[]
                h=wrapper.model.lm_head.register_forward_hook(lambda m,a,o:captured.append(o[:,-1:].detach().cpu()))
                try:off=process_item_ab(v,wrapper,None,dec['answer_ids']['A'],dec['answer_ids']['B'])
                finally:h.remove()
                value=captured[0];logits[f'{i}_{alpha}']=value
                s=score_logits(value,dec['answer_ids']['A'],dec['answer_ids']['B'],v['answer_matching_behavior'])
                official_prob=off['a_prob'] if v['answer_matching_behavior']=='(A)' else off['b_prob']
                ratio=official_prob/(off['a_prob']+off['b_prob'])
                assert abs(ratio-s['normalized_target_probability'])<1e-6
                rows.append(dict(index=i,alpha=alpha,official_normalized_probability=ratio,**s))
        wrapper.reset_all();plain=wrapper.model
        for j,w in enumerate(plain.model.layers):
            block=w.block;block.self_attn=block.self_attn.attn;plain.model.layers[j]=block
        parity=[]
        for i,v in enumerate(cases):
            dec=decision(tok,'llama2',v['question'])
            for alpha in [-1,0,1]:
                h=plain.model.layers[13].register_forward_hook(translation_hook(vec,alpha,dec['boundary']),with_kwargs=True) if alpha else None
                try:
                    ids=torch.tensor([dec['input_ids']],device='cuda')
                    value=plain(ids).logits[:,-1:].cpu()
                finally:
                    if h:h.remove()
                error=float((value-logits[f'{i}_{alpha}']).abs().max())
                assert error<=1e-5
                parity.append(dict(index=i,alpha=alpha,max_full_vocab_logit_error=error))
    torch.save(logits,out/'official_logits.pt')
    save(out/'official_scores.json',rows);save(out/'same_runtime_parity.json',parity);save(out/'prompts.json',prompts)
    finish(out,m,records=len(rows),max_same_runtime_logit_error=max(x['max_full_vocab_logit_error'] for x in parity))

def new():
    assert transformers.__version__=='5.6.2'
    from transformers import AutoModelForCausalLM
    out,m=setup('new_runtime')
    tok=tokenizer('llama2');model=place_model(AutoModelForCausalLM.from_pretrained(snapshot('llama2'),local_files_only=True,
        token=False,dtype=torch.float32,attn_implementation='eager'))
    block=model.model.layers[13]
    def forward(ids,hidden=False):
        return model(input_ids=torch.tensor([ids],device='cuda'),
            position_ids=torch.arange(len(ids),device='cuda').unsqueeze(0),use_cache=True,
            output_hidden_states=hidden,logits_to_keep=1)
    train=json.loads((HERE/'training_manifest.json').read_text())
    cached_path=SCRATCH/'diagnostics/models/llama2/training64.npz'
    cached=np.load(cached_path)['corrigible_more']
    original_acts=[];swapped_acts=[];extraction=[];cache_checks=[]
    with torch.inference_mode():
        # Re-extract all groups on the allocated hardware; compare the cached64.
        for gi,g in enumerate(train):
            for vi in [0,1]:
                v=g['views'][vi];dec=decision(tok,'llama2',v['question'])
                for label,answer in [(1,v['answer_matching_behavior']),(-1,v['answer_not_matching_behavior'])]:
                    cap=[]
                    def hook(m,a,o):
                        hidden=o[0] if isinstance(o,tuple) else o
                        cap.append(hidden[0,dec['extraction_position']].clone())
                    h=block.register_forward_hook(hook)
                    try:ret=forward(dec['completion_ids'][answer[1]],hidden=(gi==0 and vi==0 and label==1))
                    finally:h.remove()
                    if gi==0 and vi==0 and label==1:
                        assert torch.equal(cap[0],ret.hidden_states[14][0,dec['extraction_position']])
                    value=cap[0].cpu().numpy()
                    if vi==0:
                        original_acts.append(value)
                        if gi<64:
                            err=float(np.max(np.abs(value-cached[2*gi+(0 if label==1 else 1)])))
                            assert err<1e-3
                            cache_checks.append(dict(group_index=gi,label=label,max_activation_error=err))
                    else:swapped_acts.append(value)
                    extraction.append(dict(group_index=gi,group_id=g['group_id'],variant=vi,label=label,
                        answer=answer,prompt=tok.decode(dec['completion_ids'][answer[1]]),
                        token_position=dec['extraction_position'],token_id=dec['completion_ids'][answer[1]][dec['extraction_position']]))
            if gi%64==0:print('extraction',gi,flush=True)
        acts=torch.from_numpy(np.stack(original_acts));swaps=torch.from_numpy(np.stack(swapped_acts))
        assert acts.shape==swaps.shape==(512,4096)
        differences=acts[0::2]-acts[1::2];swapdiff=swaps[0::2]-swaps[1::2]
        raw={f'first{n}':differences[:n].mean(0) for n in [12,64,256]}
        raw.update({f'balanced{n}':torch.cat([differences[:n],swapdiff[:n]]).mean(0) for n in [12,256]})
        original_vector=torch.load(ORIGINAL_VECTOR,weights_only=True)
        pilot_path=SCRATCH/'pilot_v1/llama2/linear_vectors.pt'
        new12=torch.load(pilot_path,weights_only=True)['corrigible_more']
        norm_old=original_vector.norm();norm_new=new12.norm()
        def scaled(x,norm):return x*norm/x.norm()
        vectors={'original':original_vector,'new12':new12,
            'original_at_new_norm':scaled(original_vector,norm_new),
            'new12_at_original_norm':scaled(new12,norm_old),
            'new64_at_original_norm':scaled(raw['first64'],norm_old),
            'new256_at_original_norm':scaled(raw['first256'],norm_old),
            'balanced12_at_original_norm':scaled(raw['balanced12'],norm_old),
            'balanced256_at_original_norm':scaled(raw['balanced256'],norm_old)}
        assert torch.nn.functional.cosine_similarity(raw['first12'],new12,dim=0)>.999999
        stability=[]
        for name,x in vectors.items():
            stability.append(dict(vector=name,norm=float(x.norm()),
                cosine_with_original=float(torch.nn.functional.cosine_similarity(x,original_vector,dim=0)),
                cosine_with_new12=float(torch.nn.functional.cosine_similarity(x,new12,dim=0)),
                dot_with_original=float(x@original_vector),sign_convention='matching minus non-matching'))
        pairwise=[dict(first=a,second=b,cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)))
            for a,x in raw.items() for b,y in raw.items()]
        save(out/'vector_statistics.json',stability);save(out/'raw_direction_pairwise.json',pairwise)
        save(out/'raw_norms.json',{n:float(x.norm()) for n,x in raw.items()})
        save(out/'extraction_prompts.json',extraction)
        save(out/'cached_activation_parity.json',cache_checks)
        torch.save(dict(raw=raw,steering=vectors),out/'vectors.pt')
        np.savez_compressed(out/'training_activations.npz',first=acts.numpy(),swapped=swaps.numpy())
        m['cached_training_sha256']=digest(cached_path);m['pilot_vector_sha256']=digest(pilot_path)
        vectors={n:x.cuda() for n,x in vectors.items()}
        cases=json.loads((HERE/'evaluation_manifest.json').read_text())
        old_logits=torch.load(OUT/'original_runtime/official_logits.pt',weights_only=True)
        old_prompts=json.loads((OUT/'original_runtime/prompts.json').read_text())
        pilot_logits=torch.load(SCRATCH/'pilot_v1/llama2/decision_logits.pt',weights_only=True)
        rows=[];prompts=[];parity=[];pilotparity=[];full={}
        for ci,v in enumerate(cases):
            dec=decision(tok,'llama2',v['question']);prompts.append(dict(case_index=ci,**v,**dec))
            if v['dataset']=='original50':assert dec['input_ids']==old_prompts[v['group_index']]['input_ids']
            settings=[('no_hook',0,None)]+[(name,alpha,vec) for name,vec in vectors.items() for alpha in [-1,1]]
            for name,alpha,vec in settings:
                observations=[]
                def intervention(module,args,kwargs,output):
                    hidden=output[0] if isinstance(output,tuple) else output
                    positions=kwargs['position_ids'];mask=(positions>=dec['boundary']).expand(hidden.shape[:2])
                    before=hidden[mask].clone();outside=hidden[~mask].clone()
                    ret=translation_hook(vec,alpha,dec['boundary'])(module,args,kwargs,output)
                    assert torch.equal(hidden[~mask],outside)
                    delta=hidden[mask]-before
                    observations.append(dict(positions=positions[mask].cpu().tolist(),
                        activation_norm=before.norm(dim=-1).cpu().tolist(),
                        displacement=delta.norm(dim=-1).cpu().tolist(),
                        relative_displacement=(delta.norm(dim=-1)/before.norm(dim=-1)).cpu().tolist()))
                    return ret
                h=block.register_forward_hook(intervention,with_kwargs=True) if vec is not None else None
                try:logits=forward(dec['input_ids']).logits.cpu()
                finally:
                    if h:h.remove()
                s=score_logits(logits,dec['answer_ids']['A'],dec['answer_ids']['B'],v['answer_matching_behavior'])
                target_a=v['answer_matching_behavior']=='(A)'
                probs=logits[0,-1].float().softmax(-1)
                target_id=dec['answer_ids']['A' if target_a else 'B']
                p_full=float(probs[target_id]);mass=s['p_a_full_vocab']+s['p_b_full_vocab']
                assert abs(p_full/mass-s['normalized_target_probability'])<1e-6
                row=dict(case_index=ci,dataset=v['dataset'],group_id=v['group_id'],group_index=v['group_index'],
                    variant=v['variant'],target=v['answer_matching_behavior'],vector=name,alpha=alpha,
                    logit_a=float(logits[0,-1,dec['answer_ids']['A']]),logit_b=float(logits[0,-1,dec['answer_ids']['B']]),
                    logsumexp_full_vocab=float(torch.logsumexp(logits[0,-1].float(),0)),
                    full_vocab_target_probability=p_full,ab_total_probability=mass,
                    hook_calls=len(observations),hook_diagnostics=observations,**s)
                rows.append(row)
                if v['dataset']=='original50' and name in ['original','no_hook']:
                    key=f"{v['group_index']}_{alpha}"
                    ref=old_logits[key];err=float((logits-ref).abs().max())
                    oldscore=score_logits(ref,dec['answer_ids']['A'],dec['answer_ids']['B'],v['answer_matching_behavior'])
                    parity.append(dict(index=v['group_index'],alpha=alpha,max_full_vocab_logit_error=err,
                        normalized_probability_error=abs(s['normalized_target_probability']-oldscore['normalized_target_probability'])))
                    full[key]=logits
                if v['dataset']=='new_validation206' and v['group_index']<2 and name in ['no_hook','new12']:
                    mode='no_hook_no_hook' if alpha==0 else 'linear_'+('positive' if alpha>0 else 'negative')
                    rid=f"corrigible_more_{v['group_index']}_{v['variant']}_{mode}"
                    ref=pilot_logits[rid];err=float((logits-ref).abs().max());assert err<1e-3
                    refscore=score_logits(ref,dec['answer_ids']['A'],dec['answer_ids']['B'],v['answer_matching_behavior'])
                    perr=abs(s['normalized_target_probability']-refscore['normalized_target_probability']);assert perr<1e-4
                    pilotparity.append(dict(record=rid,max_full_vocab_logit_error=err,normalized_probability_error=perr))
            if ci%25==0:
                print('evaluated',ci+1,'of',len(cases),'records',len(rows),flush=True)
                save(out/'progress.json',dict(cases_completed=ci+1,total_cases=len(cases),records=len(rows)))
        save(out/'results.json',rows);save(out/'evaluation_prompts.json',prompts)
        save(out/'cross_runtime_parity.json',parity);save(out/'pilot_parity.json',pilotparity)
        torch.save(full,out/'original50_logits.pt')
        # Tolerances were fixed before inference; report measured differences.
        assert max(x['normalized_probability_error'] for x in parity)<1e-4
        assert max(x['max_full_vocab_logit_error'] for x in parity)<1e-3
    finish(out,m,records=len(rows),cases=len(cases),pilot_parity_checks=len(pilotparity),
        max_cross_runtime_logit_error=max(x['max_full_vocab_logit_error'] for x in parity),
        max_cross_runtime_probability_error=max(x['normalized_probability_error'] for x in parity))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['original','new'],required=True);a=p.parse_args()
    original() if a.stage=='original' else new()
