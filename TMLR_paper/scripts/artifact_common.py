"""Read-only historical inputs; all revision outputs live in TMLR_paper."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'TMLR_paper'
SMALL=ROOT/'results/small_lambda_positive_20261110'
TEST=ROOT/'results/small_lambda_validation_test_20261110'
LLAMA=ROOT/'results/llama2_small_lambda_positive_20261111'
NAMES={'coordinate-other-ais':'AI Coordination','corrigible-neutral-HHH':'Corrigibility','hallucination':'Hallucination','myopic-reward':'Myopic Reward','survival-instinct':'Survival Instinct','sycophancy':'Sycophancy','refusal':'Refusal'}
MODELS={'qwen3':'Qwen3-1.7B','llama3b':'Llama-3.2-3B-Instruct','llama2':'Llama-2-7B-Chat'}
WEIGHTS=[0.,.001,.005,.01,.05,.1,.25,.5,1.]
load=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def write_csv(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rs for k in r));tmp=p.with_suffix('.csv.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rs:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
 tmp.replace(p)
def table(headers,body):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in body])
def tex_table(path,headers,body,caption,label,number=None):
 esc=lambda x:str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
 lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(esc(x) for x in headers)+r' \\',r'\midrule']
 lines+=[' & '.join(esc(x) for x in r)+r' \\' for r in body]
 lines += [r'\bottomrule',r'\end{tabular}}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table*}']
 Path(path).write_text('\n'.join(lines)+'\n')
def base(model):return LLAMA if model=='llama2' else SMALL
def config(model):return load(base(model)/'experiment_config.json')
def selected(model):
 if model=='llama2':return load(LLAMA/'selected_configurations.json')['selected']
 return {r['concept']:float(r['selected_weight']) for r in load(TEST/'frozen_config.json')['selected'] if r['model']==model}
def fit_path(model,c,w):return base(model)/'fits'/model/f'{c}_lambda_{float(w)}.joblib'
def fit_class():
 sys.path.insert(0,str(ROOT/'experiments/qwen_quadratic_attenuation_20261103'))
 sys.path.insert(0,str(ROOT/'experiments/small_lambda_positive_20261110'))
 from weighted_small import AttenuatedCurveball
 return AttenuatedCurveball
SPLIT=load(ROOT/'results/qwen_quadratic_attenuation_20261103/split_manifest.json')
# Actual manifest is a row list, shared by all three models.
def split_rows(c,part):return [r for r in SPLIT if r['concept']==c and r['partition']==part]
def raw_rows(model,c,part):
 p=base(model)/'rows'/model/f'{c}_{part}.jsonl';out=[]
 for line in p.read_text().splitlines():out.extend(json.loads(line)['rows'])
 return out

def plot_style():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
 return plt

def savefig(fig,name):
 for ext in ['pdf','png']:fig.savefig(PAPER/'figures'/f'{name}.{ext}',dpi=180)
 import matplotlib.pyplot as plt
 plt.close(fig)
