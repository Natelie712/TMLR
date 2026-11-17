"""Frozen polynomial Curveball arithmetic and measured strength calibration.

Preparation only: this module does not load language models or launch a sweep.
"""
import numpy as np
import torch
from sklearn.decomposition import KernelPCA


class PolynomialCurveball:
    def __init__(self, x, labels, device='cpu'):
        x=np.asarray(x,dtype=np.float64); labels=np.asarray(labels)
        assert x.ndim==2 and set(labels)=={-1,1} and np.isfinite(x).all()
        self.mean=x.mean(0)
        self.fit=KernelPCA(n_components=min(20,len(x)-1),kernel='poly',degree=2,
            gamma=.01,coef0=1.,alpha=1.,fit_inverse_transform=True,
            eigen_solver='arpack',random_state=42,remove_zero_eig=True)
        z=self.fit.fit_transform(x-self.mean)
        raw=z[labels==1].mean(0)-z[labels==-1].mean(0)
        self.raw_latent_direction_norm=float(np.linalg.norm(raw))
        assert self.raw_latent_direction_norm>1e-12
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float64,device=device)
        self.center=tensor(self.mean);self.direction=tensor(raw/self.raw_latent_direction_norm)
        self.train=tensor(self.fit.X_fit_)
        eigen=self.fit.eigenvalues_;scaled=np.zeros_like(self.fit.eigenvectors_)
        nz=eigen>0;scaled[:,nz]=self.fit.eigenvectors_[:,nz]/np.sqrt(eigen[nz])
        self.scaled=tensor(scaled)
        self.kernel_rows=tensor(self.fit._centerer.K_fit_rows_)
        self.kernel_mean=float(self.fit._centerer.K_fit_all_)
        self.train_z=tensor(self.fit.X_transformed_fit_)
        self.dual=tensor(self.fit.dual_coef_)
        reconstructed=self.fit.inverse_transform(z)+self.mean
        self.fit_diagnostics=dict(training_rows=len(x),components=z.shape[1],
            raw_latent_direction_norm=self.raw_latent_direction_norm,
            normalized_latent_direction_norm=float(self.direction.norm()),
            mean_projection_norm=float(np.linalg.norm(z,axis=1).mean()),
            reconstruction_rmse=float(np.sqrt(np.mean((reconstructed-x)**2))),
            relative_reconstruction_l2=float(np.linalg.norm(reconstructed-x)/np.linalg.norm(x)),
            precision='float64 projection/inverse; final update cast to model activation dtype')

    @staticmethod
    def kernel(x,y):return (.01*(x@y.T)+1.)**2

    def to(self,device):
        for name in ['center','direction','train','scaled','kernel_rows','train_z','dual']:
            setattr(self,name,getattr(self,name).to(device))
        return self

    def project(self,h):
        k=self.kernel(h.to(self.center)-self.center,self.train)
        return (k-k.mean(1,keepdim=True)-self.kernel_rows+self.kernel_mean)@self.scaled

    def inverse(self,z):return self.kernel(z,self.train_z)@self.dual

    def intervention(self,h,step):
        if step==0:return h
        z=self.project(h)
        update=self.inverse(z+float(step)*self.direction)-self.inverse(z)
        result=(h.to(update)+update).to(h.dtype)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite Curveball update')
        return result

    def match_strength(self,probes,target,sign):
        """First powers-of-two bracket, 25 bisections, 1% actual RMS tolerance."""
        assert sign in [-1,1] and target>0 and len(probes)>0
        trace=[]
        def measure(magnitude):
            after=self.intervention(probes,sign*magnitude)
            delta=(after.double()-probes.double()).norm(dim=-1)
            value=float(delta.square().mean().sqrt())
            trace.append(dict(native_parameter=sign*magnitude,rms_displacement=value))
            return value
        lo=0.;hi=None
        for exponent in range(-16,25):
            candidate=2.**exponent
            if measure(candidate)>=target:hi=candidate;break
            lo=candidate
        if hi is None:raise ValueError('Predeclared strength grid cannot match the target; no fallback')
        for _ in range(25):
            mid=(lo+hi)/2
            if measure(mid)>=target:hi=mid
            else:lo=mid
        step=(lo+hi)/2;actual=measure(step)
        assert abs(actual-target)/target<=.01,'Activation-space strength matching failed'
        return dict(native_parameter=sign*step,target_rms_displacement=float(target),
            actual_rms_displacement=actual,relative_error=abs(actual-target)/target,
            native_unit_rms_displacement=measure(1.),probe_positions=len(probes),trace=trace)


def angular_candidates_from_class_means(positive,negative):
    """Streaming counterpart of existing per-sample normalized Angular means.

    Inputs are per-layer means of individually normalized training activations.
    Class means are normalized before differencing; candidates retain raw norms.
    """
    p=np.asarray(positive,dtype=np.float32);n=np.asarray(negative,dtype=np.float32)
    return p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-8)-n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-8)
