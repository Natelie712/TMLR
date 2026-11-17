"""Polynomial KPCA with centered inverse ridge regression.

The decoder can use its own dimensionless scale or reuse the forward kernel's
scale and degree. Its regularization remains independently tunable. All fitting
and inference use float64: small inverse ridge values otherwise amplify float32
cancellation.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch


def capped_group_indices(indices, group_ids, cap=5000, seed=20260913):
    """Select complete semantic groups deterministically, without exceeding cap."""
    indices = np.asarray(indices)
    if len(indices) <= cap:
        return indices
    groups = np.unique(group_ids[indices])
    np.random.default_rng(seed).shuffle(groups)
    selected, size = [], 0
    for group in groups:
        rows = indices[group_ids[indices] == group]
        if size + len(rows) <= cap:
            selected.extend(rows.tolist())
            size += len(rows)
    return np.sort(np.asarray(selected, dtype=np.int64))


@dataclass(frozen=True)
class ReconstructionConfig:
    n_components: int = 70
    forward_ratio: float = 1.0
    forward_degree: int = 2
    input_center: bool = False
    decoder_ratio: float = 0.01
    decoder_degree: int = 2
    decoder_ridge: float = 1e-6
    decoder_whiten: bool = False
    decoder_coupled: bool = False
    activation_normalize: bool = False
    quadratic_weight: float = 1.0


def polynomial_kernel(inner, gamma, degree, quadratic_weight=1.0):
    """Evaluate a degree-one or weighted degree-two dot-product kernel."""
    scaled = gamma * inner
    if degree == 1:
        return scaled + 1
    return 1 + 2 * scaled + quadratic_weight * scaled.square()


def center_kernel(train, query):
    """Center both kernels using training statistics only."""
    columns, grand = train.mean(0), train.mean()
    return (
        train - train.mean(1, keepdim=True) - columns + grand,
        query - query.mean(1, keepdim=True) - columns + grand,
    )


def normalized_error(actual, reconstructed):
    energy = actual.square().sum(1).mean()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("Reconstruction reference must have positive finite energy")
    return float((actual - reconstructed).square().sum(1).mean() / energy)


class IndependentDecoderKPCA:
    """A configurable KPCA transform/decoder usable directly by steering hooks."""

    def __init__(self, config=None, device=None):
        self.config = config or ReconstructionConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        c = self.config
        if not all(
            np.isfinite(value)
            for value in (
                c.forward_ratio,
                c.decoder_ratio,
                c.decoder_ridge,
                c.quadratic_weight,
            )
        ):
            raise ValueError("Kernel ratios and inverse regularization must be finite")
        if not isinstance(c.n_components, int):
            raise TypeError("Components must be an integer")
        if c.n_components < 1 or c.forward_ratio <= 0 or c.decoder_ratio <= 0:
            raise ValueError("Components and kernel ratios must be positive")
        if c.forward_degree not in (1, 2) or c.decoder_degree not in (1, 2):
            raise ValueError("Only degree 1 and degree 2 are supported")
        if c.decoder_ridge <= 0:
            raise ValueError("Inverse regularization must be positive")
        if c.quadratic_weight < 0:
            raise ValueError("Quadratic weight must be nonnegative")
        if c.decoder_coupled and c.decoder_whiten:
            raise ValueError("A coupled decoder cannot whiten its latent inputs")

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def prepare_activations(self, values):
        """Convert activations to the space used by every KPCA computation."""
        x = self.tensor(values)
        if self.config.activation_normalize:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return x

    def fit_forward(self, values):
        x = self.prepare_activations(values)
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Training activations must be a finite matrix")
        if self.config.n_components >= len(x):
            raise ValueError(
                "Components must be smaller than the training sample count"
            )
        self.input_mean = (
            x.mean(0) if self.config.input_center else torch.zeros_like(x[0])
        )
        self.x_fit = x - self.input_mean
        energy = self.x_fit.square().sum(1).mean()
        if energy <= 0:
            raise ValueError("Training activations must have nonzero energy")
        self.forward_gamma = self.config.forward_ratio / float(energy)
        kernel = polynomial_kernel(
            self.x_fit @ self.x_fit.T,
            self.forward_gamma,
            self.config.forward_degree,
            self.config.quadratic_weight,
        )
        self.forward_columns, self.forward_grand = kernel.mean(0), kernel.mean()
        centered, _ = center_kernel(kernel, kernel)
        eigenvalues, vectors = torch.linalg.eigh(centered)
        k = self.config.n_components
        eigenvalues, vectors = eigenvalues[-k:].flip(0), vectors[:, -k:].flip(1)
        keep = eigenvalues > max(float(eigenvalues[0]) * 1e-12, 1e-12)
        if not keep.any():
            raise ValueError("Forward kernel has no positive retained eigenvalues")
        self.eigenvalues = eigenvalues[keep]
        self.scaled_vectors = vectors[:, keep] / self.eigenvalues.sqrt()
        self.z_fit = vectors[:, keep] * self.eigenvalues.sqrt()
        return self

    def transform(self, values):
        x = self.prepare_activations(values) - self.input_mean
        kernel = polynomial_kernel(
            x @ self.x_fit.T,
            self.forward_gamma,
            self.config.forward_degree,
            self.config.quadratic_weight,
        )
        centered = (
            kernel
            - kernel.mean(1, keepdim=True)
            - self.forward_columns
            + self.forward_grand
        )
        return centered @ self.scaled_vectors

    def fit_inverse(self, values):
        x = self.prepare_activations(values)
        z = self.transform(x)
        self.decoder_scale = (
            z.std(0, unbiased=False).clamp_min(1e-12)
            if self.config.decoder_whiten
            else torch.ones_like(z[0])
        )
        self.decoder_z_fit = z / self.decoder_scale
        self.decoder_gamma = (
            self.forward_gamma
            if self.config.decoder_coupled
            else self.config.decoder_ratio
            / float(self.decoder_z_fit.square().sum(1).mean())
        )
        self.decoder_degree = (
            self.config.forward_degree
            if self.config.decoder_coupled
            else self.config.decoder_degree
        )
        kernel = polynomial_kernel(
            self.decoder_z_fit @ self.decoder_z_fit.T,
            self.decoder_gamma,
            self.decoder_degree,
            self.config.quadratic_weight,
        )
        self.decoder_columns, self.decoder_grand = kernel.mean(0), kernel.mean()
        centered, _ = center_kernel(kernel, kernel)
        self.target_mean = x.mean(0)
        self.decoder_alpha = self.config.decoder_ridge * float(torch.trace(centered))
        self.dual = torch.linalg.solve(
            centered
            + self.decoder_alpha * torch.eye(len(x), dtype=x.dtype, device=self.device),
            x - self.target_mean,
        )
        return self

    def fit(self, values):
        return self.fit_forward(values).fit_inverse(values)

    def inverse_transform(self, values):
        z = self.tensor(values) / self.decoder_scale
        kernel = polynomial_kernel(
            z @ self.decoder_z_fit.T,
            self.decoder_gamma,
            self.decoder_degree,
            self.config.quadratic_weight,
        )
        centered = (
            kernel
            - kernel.mean(1, keepdim=True)
            - self.decoder_columns
            + self.decoder_grand
        )
        return centered @ self.dual + self.target_mean

    def reconstruct(self, values, batch_size=256):
        x = self.tensor(values)
        prediction = torch.cat(
            [
                self.inverse_transform(self.transform(batch))
                for batch in x.split(batch_size)
            ]
        )
        if self.config.activation_normalize:
            prediction = prediction * x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return prediction

    def save(self, path):
        state = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in vars(self).items()
            if key not in {"device", "config"}
        }
        torch.save({"config": asdict(self.config), "state": state}, path)

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(ReconstructionConfig(**payload["config"]), device=device)
        result.__dict__.update(
            {
                key: value.to(result.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in payload["state"].items()
            }
        )
        return result


def pca_reconstruct(train, query, n_components=70):
    """Exact PCA baseline with the same data, centering, and retained dimension."""
    mean = train.mean(0)
    centered = train - mean
    eigenvalues, vectors = torch.linalg.eigh(centered @ centered.T)
    eigenvalues, vectors = eigenvalues[-n_components:], vectors[:, -n_components:]
    keep = eigenvalues > max(float(eigenvalues[-1]) * 1e-12, 1e-12)
    basis = centered.T @ vectors[:, keep] / eigenvalues[keep].sqrt()
    return ((query - mean) @ basis) @ basis.T + mean


def decoder_sweep(forward, train, validation, ratios, ridges, whitenings=(False, True)):
    """Evaluate inverse kernels without refitting the forward KPCA per ridge."""
    x = forward.prepare_activations(train)
    v = forward.prepare_activations(validation)
    z, zv = forward.z_fit, forward.transform(v)
    mean = x.mean(0)
    for whiten in whitenings:
        scale = (
            z.std(0, unbiased=False).clamp_min(1e-12)
            if whiten
            else torch.ones_like(z[0])
        )
        zz, vv = z / scale, zv / scale
        inner, cross = zz @ zz.T, vv @ zz.T
        latent_energy = float(zz.square().sum(1).mean())
        for degree in (1, 2):
            for ratio in ratios if degree == 2 else [1.0]:
                gamma = ratio / latent_energy
                kernel, query = center_kernel(
                    polynomial_kernel(
                        inner, gamma, degree, forward.config.quadratic_weight
                    ),
                    polynomial_kernel(
                        cross, gamma, degree, forward.config.quadratic_weight
                    ),
                )
                eigenvalues, vectors = torch.linalg.eigh(kernel)
                targets, queries = vectors.T @ (x - mean), query @ vectors
                trace = float(torch.trace(kernel))
                for ridge in ridges:
                    prediction = (
                        queries / (eigenvalues + ridge * trace)
                    ) @ targets + mean
                    error = normalized_error(v, prediction)
                    if not np.isfinite(error):
                        raise FloatingPointError(
                            "Non-finite inverse reconstruction error"
                        )
                    yield {
                        **asdict(forward.config),
                        "decoder_ratio": ratio,
                        "decoder_degree": degree,
                        "decoder_ridge": ridge,
                        "decoder_whiten": whiten,
                        "validation_nmse": error,
                    }


def coupled_decoder_sweep(forward, train, validation, ridges):
    """Use the exact forward gamma and degree for a centered inverse regression."""
    x = forward.prepare_activations(train)
    v = forward.prepare_activations(validation)
    z, zv = forward.z_fit, forward.transform(v)
    mean = x.mean(0)
    gamma, degree = forward.forward_gamma, forward.config.forward_degree
    kernel, query = center_kernel(
        polynomial_kernel(
            z @ z.T, gamma, degree, forward.config.quadratic_weight
        ),
        polynomial_kernel(
            zv @ z.T, gamma, degree, forward.config.quadratic_weight
        ),
    )
    eigenvalues, vectors = torch.linalg.eigh(kernel)
    targets, queries = vectors.T @ (x - mean), query @ vectors
    trace = float(torch.trace(kernel))
    for ridge in ridges:
        prediction = (queries / (eigenvalues + ridge * trace)) @ targets + mean
        error = normalized_error(v, prediction)
        if not np.isfinite(error):
            raise FloatingPointError("Non-finite coupled reconstruction error")
        yield {
            **asdict(forward.config),
            "decoder_degree": degree,
            "decoder_ridge": ridge,
            "decoder_whiten": False,
            "decoder_coupled": True,
            "validation_nmse": error,
        }
