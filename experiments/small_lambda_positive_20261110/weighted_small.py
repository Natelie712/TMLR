"""Isolated small-weight extension; arithmetic unchanged, accepted weights generalized."""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/caa_official_seven_behaviors_four_models'))
from methods_b import PolynomialCurveball
from src.kpca.independent_decoder import polynomial_kernel

class WeightedKernelPCA(KernelPCA):
    quadratic_weight=1.
    def _get_kernel(self,X,Y=None):
        if self.quadratic_weight==1.:
            return super()._get_kernel(X,Y)
        if Y is None:Y=X
        # Reuse Amir's exact weighted degree-two formula, c=1, gamma=.01.
        inner=torch.from_numpy(np.asarray(X)@np.asarray(Y).T)
        return polynomial_kernel(inner,.01,2,self.quadratic_weight).numpy()

class AttenuatedCurveball(PolynomialCurveball):
    def __init__(self,x,labels,weight,device='cpu'):
        assert np.isfinite(weight) and 0. <= weight <= 1.
        self.weight=float(weight)
        x=np.asarray(x,dtype=np.float64);labels=np.asarray(labels)
        self.mean=x.mean(0)
        self.fit=WeightedKernelPCA(n_components=min(20,len(x)-1),kernel='poly',degree=2,
            gamma=.01,coef0=1.,alpha=1.,fit_inverse_transform=True,
            eigen_solver='arpack',random_state=42,remove_zero_eig=True)
        self.fit.quadratic_weight=self.weight
        z=self.fit.fit_transform(x-self.mean)
        raw=z[labels==1].mean(0)-z[labels==-1].mean(0)
        self.raw_latent_direction_norm=float(np.linalg.norm(raw))
        assert self.raw_latent_direction_norm>1e-12
        self._initialize(raw/self.raw_latent_direction_norm,device)
        reconstructed=self.fit.inverse_transform(z)+self.mean
        self.fit_diagnostics=dict(weight=self.weight,training_rows=len(x),components=z.shape[1],
            raw_latent_direction_norm=self.raw_latent_direction_norm,unit_direction_norm=float(self.direction.norm()),
            reconstruction_rmse=float(np.sqrt(np.mean((reconstructed-x)**2))),
            relative_reconstruction_l2=float(np.linalg.norm(reconstructed-x)/np.linalg.norm(x)),
            minimum_retained_eigenvalue=float(self.fit.eigenvalues_.min()),
            maximum_retained_eigenvalue=float(self.fit.eigenvalues_.max()),
            inverse_kernel='same attenuated kernel; absolute ridge=1; no inverse-kernel centering added')

    def _initialize(self,direction,device):
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float64,device=device)
        self.center=tensor(self.mean);self.direction=tensor(direction)
        self.train=tensor(self.fit.X_fit_)
        eigen=self.fit.eigenvalues_;scaled=np.zeros_like(self.fit.eigenvectors_);nz=eigen>0
        scaled[:,nz]=self.fit.eigenvectors_[:,nz]/np.sqrt(eigen[nz])
        self.scaled=tensor(scaled);self.kernel_rows=tensor(self.fit._centerer.K_fit_rows_)
        self.kernel_mean=float(self.fit._centerer.K_fit_all_)
        self.train_z=tensor(self.fit.X_transformed_fit_);self.dual=tensor(self.fit.dual_coef_)

    def kernel(self,x,y):
        if self.weight==1.:return PolynomialCurveball.kernel(x,y)
        return polynomial_kernel(x@y.T,.01,2,self.weight)

    def state(self):
        return dict(weight=self.weight,fit=self.fit,mean=self.mean,direction=self.direction.cpu().numpy(),diagnostics=self.fit_diagnostics)

    @classmethod
    def restore(cls,state,device='cpu'):
        obj=cls.__new__(cls);obj.weight=state['weight'];obj.fit=state['fit'];obj.mean=state['mean']
        obj.fit_diagnostics=state['diagnostics'];obj._initialize(state['direction'],device)
        return obj
