"""Fixed five-point validation diagnostics; reuse compatible cached endpoints."""
import argparse,importlib.util,os,sys,time,traceback
import joblib,torch
from artifact_common import *
# Explicit file loading prevents legacy sys.path changes from selecting a wrong module.
sys.path.insert(0,str(ROOT/'experiments/small_lambda_positive_20261110'))
spec=importlib.util.spec_from_file_location('tmlr_reference_runner',ROOT/'experiments/small_lambda_positive_20261110/execute.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
GRID=[-1.,-.5,0.,.5,1.]
OUT=PAPER/'data/response_curves'

def verify_frozen():
 cfg=load(PAPER/'data/revision_config.json')
 for p,h in cfg['frozen_inputs'].items():assert sha(ROOT/p)==h,p
 return cfg

class Run(old.original.Experiment):
 def __init__(self,model):
  before=old.stage.RESULT;old.stage.RESULT=OUT
  try:old.stage.Experiment.__init__(self,model,'response-'+str(time.time_ns()))
  finally:old.stage.RESULT=before
  self.bs=1;self.concepts=list(NAMES);self.variant=config(model)['prompt_variant'];self.vectors=torch.load(base(model)/'fits'/model/'linear_vectors.pt',weights_only=True)['normalized']
  self.cbs={};self.steps={};self.reference={};self.cfg=verify_frozen();self.completed=[]
  self.manifest.update(revision_config_sha256=sha(PAPER/'data/revision_config.json'),last_token=True,validation_only=True,grid=GRID)
  save(OUT/f'{model}_location.json',dict(path=str(self.out)));self.persist('initialized')
 def persist(self,status):
  self.manifest.update(status=status,completed=self.completed,elapsed_seconds=time.time()-self.start)
  save(self.out/'manifest.json',self.manifest);save(OUT/f'{self.key}_status.json',dict(status=status,completed=self.completed,path=str(self.out)))
 def load_model(self):
  from transformers import AutoModelForCausalLM
  tc=old.stage.tc;self.tok=tc.legacy.tokenizer(self.key,tc.modelpath(self.key))
  model=AutoModelForCausalLM.from_pretrained(tc.modelpath(self.key),local_files_only=True,token=False,dtype=torch.float32,attn_implementation='eager')
  if self.key=='llama3b' and torch.cuda.get_device_properties(0).total_memory<20*1024**3:
   assert torch.cuda.device_count()==2,'Two 12GB devices required for FP32 Llama3B'
   split=len(model.model.layers)//2
   def move(v,device):
    if isinstance(v,torch.Tensor):return v.to(device)
    if isinstance(v,tuple):return tuple(move(x,device) for x in v)
    if isinstance(v,list):return [move(x,device) for x in v]
    if isinstance(v,dict):return {k:move(x,device) for k,x in v.items()}
    return v
   def transfer(device):
    def hook(module,args,kwargs):return move(args,device),move(kwargs,device)
    return hook
   model.model.embed_tokens.cuda(0)
   if hasattr(model.model,'rotary_emb'):model.model.rotary_emb.cuda(0)
   for i,layer in enumerate(model.model.layers):
    layer.to('cuda:0' if i<split else 'cuda:1')
    if i>=split:layer.register_forward_pre_hook(transfer('cuda:1'),with_kwargs=True)
   model.model.norm.cuda(0);model.model.norm.register_forward_pre_hook(transfer('cuda:0'),with_kwargs=True);model.lm_head.cuda(0)
   self.manifest['placement']=dict(type='two-GPU FP32 layer split',split_block=split)
  else:
   if self.key=='llama2':assert torch.cuda.get_device_properties(0).total_memory>40*1024**3
   model.cuda();self.manifest['placement']='single GPU FP32'
  self.model=model.eval();self.block=self.model.model.layers[self.block_index]
  self.manifest.update(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],snapshot=tc.modelpath(self.key))
  self.persist('loaded')
 def tasks_for(self,c):
  from runtime_b import data,task
  src=data(c,'generate')
  return [dict(task(self.tok,self.key,c,r['index'],src[r['index']],self.variant,'generate'),group_id=r['group_id'],partition='validation') for r in split_rows(c,'validation')]
 def run(self):
  self.load_model();CB=fit_class();params=load(base(self.key)/'fits'/self.key/'positive_steps.json')['steps'];selection=selected(self.key);replay=[]
  for c in NAMES:
   weights=sorted({0.,selection[c]});self.cbs[c]={w:CB.restore(joblib.load(fit_path(self.key,c,w)),'cuda') for w in weights};step=float(params[c]['1']);self.steps[c]={str(a):step*a for a in GRID}
   methods=['Linear']+[f'lambda_{w}' for w in weights];tasks=self.tasks_for(c);cache={(r['index'],r['method']):r for r in raw_rows(self.key,c,'validation')}
   # Check a short, deterministic replay before reusing any probabilities.
   for t in sorted(tasks,key=lambda x:x['index'])[:2]:
    self.reference={}
    for method in ['No-hook']+methods:
     a=0. if method=='No-hook' else 1.;new=self.condition([t],method,a)[0];ref=cache[t['index'],method]
     assert new['candidate_token_ids']==ref['candidate_token_ids'] and new['target']==ref['target']
     err=abs(new['normalized_target_probability']-ref['normalized_target_probability']);assert err<1e-4,(self.key,c,method,err)
     replay.append(dict(model=self.key,concept=c,index=t['index'],method=method,probability_error=err))
    basep=self.condition([t],'No-hook',0.)[0]['normalized_target_probability']
    for w in weights:
     z=self.condition([t],f'lambda_{w}',0.,step_override=0.)[0]['normalized_target_probability'];assert z==basep
   save(self.out/'replay_checks.json',dict(passed=True,rows=replay))
   path=OUT/self.key/f'{c}.jsonl';path.parent.mkdir(parents=True,exist_ok=True);done=set()
   if path.exists():
    good=0
    with path.open('rb') as f:
     for line in f:
      try:obj=json.loads(line)
      except json.JSONDecodeError:break
      assert obj['revision_sha256']==sha(PAPER/'data/revision_config.json')
      for r in obj['rows']:
       key=(r['index'],r['method'],r['multiplier']);assert key not in done;done.add(key)
      good+=len(line)
    if good<path.stat().st_size:
     with path.open('r+b') as f:f.truncate(good)
   with path.open('a') as f:
    for t in tasks:
     self.reference={}
     for method in methods:
      for a in GRID:
       key=(t['index'],method,a)
       if key in done:continue
       if a in [0.,1.]:
        rr=dict(cache[t['index'],'No-hook' if a==0 else method]);rr.update(provenance='reused verified frozen validation endpoint',method=method)
       else:
        rr=self.condition([t],method,a)[0];rr['provenance']='new fixed-grid validation GPU inference'
       rr.update(multiplier=a,model=self.key,selected_weight=selection[c],native_parameter=0. if a==0 else a if method=='Linear' else a*step,partition='validation',question_group=t['group_id'],true_nohook=a==0)
       if a==0:assert not rr['intervention_triggered']
       f.write(json.dumps(dict(revision_sha256=sha(PAPER/'data/revision_config.json'),rows=[rr]))+'\n');f.flush();os.fsync(f.fileno());done.add(key)
     if t['index']%20==0:print(json.dumps(dict(model=self.key,concept=c,records=len(done),expected=len(tasks)*len(methods)*5)),flush=True)
   assert len(done)==len(tasks)*len(methods)*5
   for cb in self.cbs[c].values():cb.to('cpu')
   self.completed.append(c);self.persist('running')
  verify_frozen();self.persist('complete')

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);a=p.parse_args();verify_frozen();OUT.mkdir(exist_ok=True)
 status=OUT/f'{a.model}_status.json'
 if status.exists() and load(status)['status']=='complete':print('Already complete; no inference repeated.');return
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ex=Run(a.model)
 try:
  with torch.inference_mode():ex.run()
 except Exception:
  ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise
if __name__=='__main__':main()
