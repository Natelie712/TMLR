"""Small CPU correctness checks; no model inference or scientific parameter tuning."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/small_lambda_positive_20261110'),
             str(ROOT/'experiments/caa_official_seven_behaviors_four_models')]
from weighted_small import AttenuatedCurveball
from scoring import score

rng=np.random.default_rng(42)
x=rng.normal(size=(50,24)); labels=np.r_[np.ones(25),-np.ones(25)]
x[:,0]+=labels*.5
h=torch.tensor(rng.normal(size=(4,24)),dtype=torch.float64)
for weight in [0.,.001,.01,.5,1.]:
    cb=AttenuatedCurveball(x,labels,weight)
    z=cb.project(h)
    np.testing.assert_allclose(z.numpy(),cb.fit.transform(h.numpy()-cb.mean),rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(cb.inverse(z).numpy(),cb.fit.inverse_transform(z.numpy()),rtol=1e-8,atol=1e-9)
    assert torch.equal(cb.intervention(h,0),h)
    assert torch.isfinite(cb.intervention(h,1)).all()
    assert (cb.intervention(h,1)-h).norm()>0
logits=torch.tensor([2.,3.,4.,-1.])
ab=score(logits,{'A':0,'B':1},'A')
assert abs(ab['normalized_target_probability']-ab['scheme_A_stable_probability'])<1e-14
assert abs(ab['normalized_target_probability']-ab['scheme_A_original_probability'])<1e-7
multi=score(logits,{'A':0,'B':1,'C':2},'C')
assert multi['normalized_target_probability']>0
assert abs(multi['normalized_target_probability']-float(logits[:3].double().softmax(0)[2]))<1e-14
print('PASS: weighted KPCA projection/inverse parity, zero-step identity, finite nonzero update, A/B equivalence and non-A/B target scoring.')
