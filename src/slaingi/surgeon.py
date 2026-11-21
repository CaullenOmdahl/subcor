"""Contrastive Weight-Space Surgeon (zero-gradient structural defect ablation).

Inspired by MIT CSAIL's "Sparks of Weight-Space Programming" (Experiment 8):
given activation snapshots from verified failing/looping trajectories versus
clean passing trajectories, this module:

1. Scores each instrumented layer by normalized contrastive separation between
   failing and clean activation distributions.
2. Isolates the principal failure subspace U_k (rank-k orthonormal directions)
   directly from contrastive activation residuals—zero gradient descent.
3. Synthesizes an exact low-rank ablation weight delta:

       Delta W = -alpha * U_k @ (U_k.T @ W_base)

   so that (W_base + Delta W) @ h = (I - alpha * U_k @ U_k.T) @ W_base @ h,
   suppressing projection along the offending direction while preserving all
   orthogonal clean feature subspaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slaingi.expansion import LowRankAdapter


@dataclass(frozen=True)
class LayerDiagnosis:
    """Contrastive activation diagnostic summary for one candidate layer."""

    layer_name: str
    separation_score: float
    mean_diff_norm: float
    clean_spread: float
    directions: np.ndarray  # shape (dim, rank), orthonormal columns
    explained_ratio: float


@dataclass(frozen=True)
class SurgicalPatch:
    """Zero-gradient rank-k ablation delta targeting a specific layer."""

    layer_name: str
    adapter: LowRankAdapter
    diagnosis: LayerDiagnosis
    alpha: float
    modified_fraction_rank: float


class WeightSpaceSurgeon:
    """Analytical, zero-gradient weight-space auditor and patch synthesizer."""

    def __init__(self, *, eps: float = 1e-8) -> None:
        self._eps = eps

    def diagnose_layer(
        self,
        layer_name: str,
        failing_activations: np.ndarray,
        clean_activations: np.ndarray,
        *,
        rank: int = 1,
    ) -> LayerDiagnosis:
        """Compute contrastive separation score and failure subspace for one layer."""
        fail = np.asarray(failing_activations, dtype=np.float64)
        clean = np.asarray(clean_activations, dtype=np.float64)
        if fail.ndim != 2 or clean.ndim != 2:
            raise ValueError("Activations must be 2D arrays of shape (n_samples, dim)")
        if fail.shape[1] != clean.shape[1]:
            raise ValueError(
                f"Activation dimension mismatch: fail={fail.shape[1]}, clean={clean.shape[1]}"
            )
        dim = fail.shape[1]
        if rank <= 0 or rank > dim:
            raise ValueError(f"rank must be in [1, {dim}], got {rank}")

        mu_fail = np.mean(fail, axis=0)
        mu_clean = np.mean(clean, axis=0)
        diff = mu_fail - mu_clean
        mean_diff_norm = float(np.linalg.norm(diff))

        clean_centered = clean - mu_clean
        clean_spread = float(np.sqrt(np.mean(np.sum(clean_centered**2, axis=1)) + self._eps))

        # Normalized separation score (Fisher-like SNR across layer dimensions).
        separation_score = mean_diff_norm / (clean_spread + self._eps)

        # Build contrastive basis:
        # Column 0 is the normalized mean-difference direction (if non-zero),
        # supplemented by leading SVD directions of (fail - mu_clean) projected
        # orthogonal to previous columns.
        residuals = fail - mu_clean
        basis_cols: list[np.ndarray] = []

        if mean_diff_norm > self._eps:
            d0 = diff / mean_diff_norm
            basis_cols.append(d0)

        if len(basis_cols) < rank:
            rem = residuals.copy()
            for col in basis_cols:
                rem = rem - np.outer(rem @ col, col)
            _, s_vals, vt = np.linalg.svd(rem, full_matrices=False)
            for idx in range(vt.shape[0]):
                if len(basis_cols) >= rank:
                    break
                if s_vals[idx] <= self._eps:
                    break
                cand = vt[idx]
                for prev in basis_cols:
                    cand = cand - float(np.dot(cand, prev)) * prev
                norm_c = float(np.linalg.norm(cand))
                if norm_c > self._eps:
                    basis_cols.append(cand / norm_c)

        if not basis_cols:
            # Degenerate zero-difference case: fall back to canonical unit vectors.
            directions = np.eye(dim, rank, dtype=np.float64)
            explained_ratio = 0.0
        else:
            directions = np.column_stack(basis_cols)
            total_energy = float(np.sum(residuals**2)) + self._eps
            proj_energy = float(np.sum((residuals @ directions) ** 2))
            explained_ratio = min(1.0, proj_energy / total_energy)

        return LayerDiagnosis(
            layer_name=layer_name,
            separation_score=separation_score,
            mean_diff_norm=mean_diff_norm,
            clean_spread=clean_spread,
            directions=directions,
            explained_ratio=explained_ratio,
        )

    def rank_layers(
        self,
        layer_activations: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        rank: int = 1,
    ) -> list[LayerDiagnosis]:
        """Diagnose all instrumented layers and sort descending by separation score."""
        diagnoses = [
            self.diagnose_layer(name, fail_act, clean_act, rank=rank)
            for name, (fail_act, clean_act) in layer_activations.items()
        ]
        diagnoses.sort(key=lambda d: d.separation_score, reverse=True)
        return diagnoses

    def synthesize_ablation_patch(
        self,
        diagnosis: LayerDiagnosis,
        w_base: np.ndarray,
        *,
        adapter_name: str,
        alpha: float = 1.0,
    ) -> SurgicalPatch:
        """Synthesize a zero-gradient rank-k ablation adapter for w_base.

        Computes Delta W = -alpha * U @ (U.T @ W_base), stored natively as
        LowRankAdapter with:
            B = -alpha * U          shape (d_out, rank)
            A = U.T @ W_base        shape (rank, d_in)
        so Delta W = B @ A without modifying w_base in place.
        """
        if not (0.0 < alpha <= 2.0):
            raise ValueError(f"alpha should be in (0.0, 2.0], got {alpha}")
        w = np.asarray(w_base, dtype=np.float64)
        u = diagnosis.directions
        if w.shape[0] != u.shape[0]:
            raise ValueError(
                f"Layer '{diagnosis.layer_name}' output dim {w.shape[0]} "
                f"does not match diagnosis direction dim {u.shape[0]}"
            )

        b_mat = -alpha * u  # (d_out, rank)
        a_mat = u.T @ w  # (rank, d_in)
        adapter = LowRankAdapter(
            name=adapter_name,
            a=a_mat,
            b=b_mat,
            scale=1.0,
            binary_ste=False,
            frozen=True,
        )
        rank_fraction = u.shape[1] / min(w.shape[0], w.shape[1])
        return SurgicalPatch(
            layer_name=diagnosis.layer_name,
            adapter=adapter,
            diagnosis=diagnosis,
            alpha=alpha,
            modified_fraction_rank=rank_fraction,
        )
