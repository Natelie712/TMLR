"""Freeze cache provenance and diagnostic design before any response inference."""
from artifact_common import *

def main():
 path=PAPER/'data/revision_config.json'
 if path.exists():
  for p,h in load(path)['frozen_inputs'].items():assert sha(ROOT/p)==h,p
  print('Existing revision freeze verified.');return
 sources={ROOT/'paper/_NeurIPS26__kernelpca_steering.pdf',ROOT/'results/qwen_quadratic_attenuation_20261103/split_manifest.json',TEST/'frozen_config.json',LLAMA/'selected_configurations.json',PAPER/'scripts/artifact_common.py',PAPER/'scripts/response_gpu.py'}
 for model in MODELS:
  cfg=config(model)
  for p,h in cfg['frozen_inputs'].items():assert sha(ROOT/p)==h,p
  sources.add(base(model)/'experiment_config.json')
  folder=Path(cfg['activation_sources'][model]);sources.update(folder/n for n in ['manifest.json','training_prompts.jsonl','extraction_index_checks.json'])
  sources.update(folder/f'train_activations_{c}.pt' for c in NAMES)
  sources.update(base(model)/'fits'/model/n for n in ['linear_vectors.pt','positive_steps.json'])
  sources.update(fit_path(model,c,w) for c in NAMES for w in WEIGHTS)
  sources.update(base(model)/'rows'/model/f'{c}_validation.jsonl' for c in NAMES)
 cfg=dict(models={m:config(m)['models'][m] for m in MODELS},selected={m:selected(m) for m in MODELS},response_grid=[-1.,-.5,0.,.5,1.],response_partition='validation',reference_endpoints=[0.,1.],curveball_step='Signed multiplier times frozen positive native step, shared across weights. Not displacement matching.',representation_fitting='reuse unchanged fit-only objects',no_test_tuning=True,exploratory_fixed_benchmark=True,geometry=dict(cluster_count=8,random_group_partitions=500,bootstrap_resamples=1000,permutation_resamples=1999,extra_nonselected_weight=.01,seed=20261111),frozen_inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)})
 save(path,cfg);print(json.dumps(dict(stage='frozen',protected_files=len(sources))),flush=True)
if __name__=='__main__':main()
