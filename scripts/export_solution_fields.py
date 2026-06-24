#!/usr/bin/env python3
"""Export solution-field comparison arrays for ONE in-domain front_layer test sample.

For a single representative in-domain test parameter this writes a CSV with the
1D grid and four solution fields:
  - u_fom                : full-order (high-order) solve, the FOM / HOS reference
  - u_manifold_galerkin  : Manifold Galerkin ROM reconstruction
  - u_manifold_lspg      : Manifold-LSPG ROM reconstruction
  - u_pod_lspg           : POD-LSPG ROM reconstruction

It reuses an already-trained offline artifact (decoder + scaler) for the
manuscript's ``reference_front_layer`` in-domain config; NO retraining is
performed at export time.  
"""
from __future__ import annotations

import os

# Pin every BLAS/OpenMP backend to a single thread BEFORE numpy/torch import so
# the offline-artifact evaluation and the manual single-sample replication are
# bit-reproducible run-to-run (multi-threaded float reductions otherwise perturb
# the per-sample errors at ~1e-4, which can flip near-median sample selection).
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mgp.benchmark import (  # noqa: E402
    BenchmarkConfig,
    _instantiate_autoencoder,
    _instantiate_scaler,
    _projected_residual_norm,
    _relative_l2,
    _resolve_test_parameters,
    _select_warm_start_state,
    evaluate_benchmark_from_offline_artifact,
    np_to_torch,
)
from mgp.manifold_rom import ManifoldGalerkinROM, ManifoldLSPGROM  # noqa: E402
from mgp.pod_rom import PODGalerkinROM, PODLSPGROM  # noqa: E402
from mgp.problems import create_problem  # noqa: E402

# The manuscript's reference_front_layer in-domain artifact (front_layer,
# latent_dim=2, pod_dim=2, conv1d, 250-epoch preset run at 60 epochs,
# train_size=24, residual_penalty 0.02, ambient_residual_penalty 0.03,
# deterministic publication mode).  This is the SAME offline-artifact schema the
# paper's solver-tradeoff study consumes, regenerated deterministically from the
# exact config embedded in artifacts/solver_tradeoff_expanded_study.json (the
# original cache file cache/offline_artifact_58f33a90bb645c00.pt is no longer on
# disk).  Reused as-is; the decoder is NOT retrained at export time.
#
# Seed 5 is chosen because its in-domain aggregate state errors are the closest
# single-seed match to the published 10-seed reference_front_layer means
# (MG 0.1612 / Manifold-LSPG 0.8433 / POD-LSPG 0.2821): this artifact yields
# MG 0.129 / Manifold-LSPG 0.792 / POD-LSPG 0.279 with Manifold-LSPG converging
# on every in-domain sample.  Manifold-LSPG is by a wide margin the weakest
# solver here, matching the paper's reference_front_layer ranking.  The rebuild
# was validated against the published per-seed summary: regenerating seed 0
# reproduces its solver_tradeoff_expanded_study.json means exactly
# (MG 0.1826 / Manifold-LSPG 1.0807 / POD-LSPG 0.2935), confirming the
# deterministic reconstruction is faithful.
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "reference_front_layer_in_domain_offline_artifact.pt"
)
OUTPUT_CSV = REPO_ROOT / "artifacts" / "solution_fields_front_layer.csv"


def _config_from_artifact(artifact: dict) -> BenchmarkConfig:
    """Rebuild a BenchmarkConfig from the artifact, tolerant to schema drift."""
    valid = {f.name for f in fields(BenchmarkConfig)}
    payload = {k: v for k, v in artifact["config"].items() if k in valid}
    # Tuple-typed fields are stored as lists in the artifact JSON-like dict.
    for key in ("hidden_dims", "decoder_hidden_dims"):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    config = BenchmarkConfig(**payload)
    # We hand the artifact directly to the evaluator, so disable cache lookups,
    # and force the in-domain regime (the headline comparison).
    config.use_offline_cache = False
    config.refresh_offline_cache = False
    config.test_regime = "in_domain"
    return config


def main() -> int:
    artifact_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARTIFACT
    if not artifact_path.exists():
        raise FileNotFoundError(f"offline artifact not found: {artifact_path}")

    print(f"Loading offline artifact (no retraining): {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    config = _config_from_artifact(artifact)
    print(
        f"  problem={config.problem_name} regime={config.test_regime} "
        f"latent_dim={config.latent_dim} pod_dim={config.pod_dim} "
        f"arch={config.architecture_name} epochs={config.autoencoder_epochs} "
        f"train_size={config.train_size}"
    )

    # ---- Reference metrics via the benchmark's own evaluation path (no training).
    # Used to (a) pick a representative sample and (b) cross-check our states.
    print("Evaluating benchmark from artifact to obtain reference per-case metrics ...")
    result = evaluate_benchmark_from_offline_artifact(artifact, config)
    cases = result["cases"]
    test_parameters = np.asarray(result["test_parameters"], dtype=np.float64)

    # Representative = the Manifold-LSPG-CONVERGED in-domain sample at the median
    # of the Manifold-LSPG relative-state-error distribution.  We key on
    # Manifold-LSPG (the headline weak solver) and require its convergence so the
    # displayed field is a real solve rather than a divergence artifact; choosing
    # the median (not the best case) keeps the comparison representative rather
    # than flattering Manifold-LSPG.
    #
    # We select the upper-median ORDER STATISTIC (stable-sorted position n//2)
    # rather than argmin(|err - median|).  For an even count the two central
    # samples are exactly equidistant from the arithmetic median, so the
    # argmin tie is decided by sub-precision float jitter and is NOT
    # reproducible run-to-run; the order statistic is stable because the two
    # central errors are well separated in value even when their distances tie.
    mgp_lspg_errs = np.array([c["mgp_lspg_error_l2"] for c in cases])
    mgp_lspg_conv = np.array([bool(c["mgp_lspg_converged"]) for c in cases])
    converged_idx = np.where(mgp_lspg_conv)[0]
    if converged_idx.size == 0:
        raise RuntimeError("no Manifold-LSPG-converged in-domain sample available")
    median_err = float(np.median(mgp_lspg_errs[converged_idx]))
    order = converged_idx[np.argsort(mgp_lspg_errs[converged_idx], kind="stable")]
    rep_idx = int(order[len(order) // 2])
    ref_case = cases[rep_idx]
    print(
        f"Representative in-domain sample index={rep_idx} of {len(cases)} "
        f"({converged_idx.size} Manifold-LSPG-converged; median M-LSPG err={median_err:.4f}); "
        f"mu={np.round(test_parameters[rep_idx], 5).tolist()}"
    )
    print(
        "  per-solver relative state error at this sample: "
        f"MG={ref_case['mgp_error_l2']:.4f} "
        f"POD-LSPG={ref_case['lspg_error_l2']:.4f} "
        f"Manifold-LSPG={ref_case['mgp_lspg_error_l2']:.4f}"
    )

    # ---- Rebuild the exact same objects the evaluator builds, then replicate the
    # single-sample solve to recover the reconstructed STATE fields.
    problem = create_problem(artifact["problem_name"])
    autoencoder = _instantiate_autoencoder(artifact)
    scaler = _instantiate_scaler(artifact)
    train_parameters = np.asarray(artifact["train_parameters"], dtype=np.float64)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)

    solver_kwargs = {
        "solver_tol": config.nonlinear_solver_tol,
        "solver_max_iter": config.nonlinear_solver_max_iter,
        "solver_line_search_steps": config.nonlinear_solver_line_search_steps,
        "solver_min_step_scale": config.nonlinear_solver_min_step_scale,
    }
    mgp_rom = ManifoldGalerkinROM(autoencoder, scaler, problem, **solver_kwargs)
    mgp_lspg_rom = ManifoldLSPGROM(autoencoder, scaler, problem, **solver_kwargs)
    pod_rom = PODGalerkinROM.fit(train_snapshots, reduced_dim=config.pod_dim, problem=problem, **solver_kwargs)
    lspg_rom = PODLSPGROM.fit(train_snapshots, reduced_dim=config.pod_dim, problem=problem, **solver_kwargs)

    # Confirm we resolve the SAME mu as the reference case (alignment guard).
    _, resolved_test = _resolve_test_parameters(problem, config)
    mu = np.asarray(resolved_test[rep_idx], dtype=np.float64)
    assert np.allclose(mu, test_parameters[rep_idx], atol=1e-12), "test-parameter misalignment"

    # FOM / high-order solve (the HOS reference).
    full = problem.solve_full(mu)
    full_state = full.solution.detach().cpu().numpy()
    print(f"  FOM solve: converged={full.converged} residual_norm={full.residual_norm:.3e}")

    # Warm start exactly as the benchmark (mean_train here).
    warm_state = _select_warm_start_state(
        config.warm_start_strategy, mu, train_parameters, train_snapshots
    )
    initial_latent = mgp_rom.encode(warm_state)
    initial_lspg = lspg_rom.project_state(warm_state)

    # Manifold Galerkin.
    mgp = mgp_rom.solve(mu, initial=initial_latent)
    mgp_state = mgp_rom.decode(mgp.solution.detach().cpu().numpy())

    # Manifold-LSPG (same MGP-encoded initial latent, per the benchmark).
    mgp_lspg = mgp_lspg_rom.solve(mu, initial=initial_latent)
    mgp_lspg_state = mgp_lspg_rom.decode(mgp_lspg.solution.detach().cpu().numpy())

    # POD-LSPG.
    lspg = lspg_rom.solve(mu, initial=initial_lspg)
    lspg_state = lspg_rom.reconstruct(lspg.solution.detach().cpu().numpy())

    # ---- Provenance cross-check: our replicated errors must match the benchmark's.
    rep_mgp = _relative_l2(full_state, mgp_state)
    rep_mgp_lspg = _relative_l2(full_state, mgp_lspg_state)
    rep_pod_lspg = _relative_l2(full_state, lspg_state)
    print("Cross-check replicated vs benchmark per-case relative L2:")
    print(f"  manifold_galerkin: {rep_mgp:.8e} vs {ref_case['mgp_error_l2']:.8e}")
    print(f"  manifold_lspg    : {rep_mgp_lspg:.8e} vs {ref_case['mgp_lspg_error_l2']:.8e}")
    print(f"  pod_lspg         : {rep_pod_lspg:.8e} vs {ref_case['lspg_error_l2']:.8e}")
    assert abs(rep_mgp - ref_case["mgp_error_l2"]) < 1e-6, "MGP state mismatch vs benchmark"
    assert abs(rep_mgp_lspg - ref_case["mgp_lspg_error_l2"]) < 1e-6, "MGP-LSPG state mismatch"
    assert abs(rep_pod_lspg - ref_case["lspg_error_l2"]) < 1e-6, "POD-LSPG state mismatch"
    for name, conv in (
        ("manifold_galerkin", mgp.converged),
        ("manifold_lspg", mgp_lspg.converged),
        ("pod_lspg", lspg.converged),
    ):
        if not conv:
            print(f"  WARNING: {name} did not report convergence for this sample")

    # ---- Assemble full-domain fields including homogeneous Dirichlet boundaries.
    # The state vectors are interior nodes (num_nodes-2); boundaries are u=0.
    # Appending the zero boundary rows leaves every relative-L2 above unchanged.
    x = problem.nodes.detach().cpu().numpy()  # (num_nodes,)

    def _with_boundaries(interior: np.ndarray) -> np.ndarray:
        full = np.zeros(problem.num_nodes, dtype=np.float64)
        full[1:-1] = interior
        return full

    u_fom = _with_boundaries(full_state)
    u_mgp = _with_boundaries(mgp_state)
    u_mgp_lspg = _with_boundaries(mgp_lspg_state)
    u_pod_lspg = _with_boundaries(lspg_state)

    columns = {
        "x": x,
        "u_fom": u_fom,
        "u_manifold_galerkin": u_mgp,
        "u_manifold_lspg": u_mgp_lspg,
        "u_pod_lspg": u_pod_lspg,
    }
    data = np.column_stack(list(columns.values()))
    assert np.all(np.isfinite(data)), "non-finite values present"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(columns.keys())
    np.savetxt(OUTPUT_CSV, data, delimiter=",", header=header, comments="", fmt="%.10g")
    print(f"\nWrote {OUTPUT_CSV} ({data.shape[0]} rows, {data.shape[1]} columns)")

    # ---- Report: CSV head + per-column min/max + L2 errors vs FOM.
    print("\nCSV head:")
    print(header)
    for row in data[:6]:
        print(",".join(f"{v:.6g}" for v in row))

    print("\nPer-column min/max:")
    for name, col in columns.items():
        print(f"  {name:22s} min={col.min():+.6e} max={col.max():+.6e}")

    front_sign_changes = int(np.sum(u_fom[:-1] * u_fom[1:] < 0.0))
    print(f"\nFOM field sign changes (front crossings): {front_sign_changes}")

    print("\nRelative L2 error of each ROM vs FOM (full field):")
    for name, col in (
        ("manifold_galerkin", u_mgp),
        ("manifold_lspg", u_mgp_lspg),
        ("pod_lspg", u_pod_lspg),
    ):
        print(f"  {name:22s} {_relative_l2(u_fom, col):.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
