#!/usr/bin/env python
"""
Summarize IC-generalization simulation results across n_hidden and istride runs.

For each experiment folder:
  - 10 checkpoint files are treated as "train" ICs (each model evaluated on its own IC)
  - the next 10 checkpoint IC indices are treated as "test" ICs (each evaluated with
    all 10 train models; MSE averaged over models)

Produces:
  <out_dir>/summary_results.csv
  <out_dir>/plot_vs_n_params.{pdf,png}
  <out_dir>/plot_vs_istride.{pdf,png}
  <out_dir>/plot_timeseries_nhid.{pdf,png}
  <out_dir>/plot_timeseries_istride.{pdf,png}

Usage:
    python summarize_scaling.py \
        --outputs_dir outputs/outputs \
        --data_dir    ../Thesis/datasets/gpe_simulations/ \
        --vae_checkpoint ../Thesis/outputs/vae/checkpoints/best_checkpoint.pt \
        --out_dir outputs/scaling_summary
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import GPEDataset
from src.dgpe_nn import DGPEModule
from src.models.vae import VAE
from DGPE.GPElib.dynamics_generator import DynamicsGenerator

# ── Experiment registry ──────────────────────────────────────────────────────
# (folder_name, n_hidden, istride)
EXPERIMENTS = [
    ("ic_generalization_istride250_nhid_32",  32,  250),
    ("ic_generalization_istride250_nhid_64",  64,  250),
    ("ic_generalization_istride250_nhid_128", 128, 250),
    ("ic_generalization_istride250_nhid_256", 256, 250),
    ("ic_generalization_istride10",           128,  10),
    ("ic_generalization_istride50",           128,  50),
    ("ic_generalization_istride100",          128, 100),
    ("ic_generalization_istride150",          128, 150),
]

# Reference values for the two sweep plots
NHID_SWEEP_ISTRIDE = 250   # istride fixed while n_hidden varies
ISTRIDE_SWEEP_NHID = 128   # n_hidden fixed while istride varies

# Fixed architecture / data parameters
LATENT_DIM = 128
V          = 1000          # 10 × 10 × 10 lattice
N_IN       = 1 + LATENT_DIM * 2   # 257  (time + z0)
N_OUT      = 2 * V                 # 2000 (Re + Im fields)
STEP       = 0.001
N_STEPS    = 5000
TRAIN_FRAC = 0.75          # interpolation / extrapolation boundary

N_TRAIN = 50   # number of train-IC checkpoints to use per experiment
N_TEST  = 50   # number of test ICs to evaluate per experiment


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_dgpe() -> DynamicsGenerator:
    dgpe = DynamicsGenerator(
        N_part_per_well=1.0, W=0, disorder_seed=53,
        N_wells=(10, 10, 10), dimensionality=3, anisotropy=1.0,
        threshold_XY_to_polar=0.25, J=1, beta=1.0,
        integration_method="RK45", rtol=1e-8, atol=1e-8,
        smooth_quench_to_room=True, reset_steps_duration=5,
        calculation_type="lyap_save_all", integrator="scipy",
        time=51, step=STEP, t_steps=N_STEPS, gamma=1.0, quenching_gamma=1.0,
    )
    dgpe.step   = STEP
    dgpe.n_steps = N_STEPS
    dgpe.icurr  = 0
    dgpe.inext  = 1
    return dgpe


def encode_ic(vae: VAE, X0: torch.Tensor, Y0: torch.Tensor, device) -> torch.Tensor:
    psi0 = torch.stack((X0.flatten(), Y0.flatten())).flatten().to(device)
    mu, logvar = vae.encode(psi0)
    return torch.stack((mu, logvar)).flatten()


def load_trajectory(dataset: GPEDataset, ic_idx: int, vae: VAE, device):
    """Return (X_field, Y_field, z0) for a single IC index."""
    psi, _ = dataset[int(ic_idx)]
    X_field = psi[0].numpy()   # (Nx, Ny, Nz, T)
    Y_field = psi[1].numpy()
    X0, Y0  = psi[0, :, :, :, 0], psi[1, :, :, :, 0]
    z0 = encode_ic(vae, X0, Y0, device)
    return X_field, Y_field, z0


def evaluate_batched(model: nn.Module, z0: torch.Tensor,
                     X_flat: np.ndarray, Y_flat: np.ndarray,
                     time_indices: np.ndarray, step: float,
                     device) -> np.ndarray:
    """
    Single batched forward pass over all time_indices.

    X_flat, Y_flat : pre-reshaped arrays of shape (V, T).

    Returns per-timestep MSE as a 1-D numpy array of length len(time_indices).
    Use .mean() for a scalar and keep the array for timeseries plots.
    """
    n = len(time_indices)
    t_batch = torch.tensor(step * time_indices, dtype=torch.float32, device=device).unsqueeze(1)
    z0_exp  = z0.detach().unsqueeze(0).expand(n, -1)
    inp     = torch.cat([t_batch, z0_exp], dim=1)          # (n, N_in)

    gt = torch.tensor(
        np.concatenate([X_flat[:, time_indices].T, Y_flat[:, time_indices].T], axis=1),
        dtype=torch.float32, device=device,
    )                                                       # (n, 2V)

    with torch.no_grad():
        pred = model(inp)                                   # (n, 2V)
        per_ts_mse = ((pred - gt) ** 2).mean(dim=1).cpu().numpy()

    return per_ts_mse


def load_model(ckpt_path: Path, dgpe: DynamicsGenerator,
               n_hidden: int, device) -> nn.Module:
    model = DGPEModule(dgpe, N_IN, N_OUT, n_hidden=n_hidden).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_model_valid(model: nn.Module) -> bool:
    """Return False if any parameter contains NaN or Inf (diverged training)."""
    return all(torch.isfinite(p).all().item() for p in model.parameters())


# ── Per-experiment evaluation ─────────────────────────────────────────────────

def run_experiment(exp_dir: str, n_hidden: int, istride: int,
                   dgpe: DynamicsGenerator, vae: VAE,
                   dataset: GPEDataset, device,
                   train_ckpt_names: list = None,
                   test_ic_indices_override: list = None) -> dict:
    """
    Evaluate one experiment.  Returns a flat dict with:
      - median/std MSE for each of the 4 tasks: (train|test) × (interp|extrap)
      - time_frac    : 1-D array of time fractions in [0, 1]
      - mse_curve_*  : median over ICs of the normalised per-timestep MSE curve
    Diverged models (NaN/Inf weights or predictions) are silently skipped.
    All time points are used for both scalar metrics and timeseries curves.

    train_ckpt_names        : list of checkpoint filenames (e.g. 'pinn_ic_52.pt')
                              to use as train ICs; if None auto-detects top N_TRAIN.
    test_ic_indices_override: list of IC indices (int) to use as test ICs;
                              if None auto-detects next N_TEST checkpoints.
    """
    ckpt_dir = Path(exp_dir) / "checkpoints"

    if train_ckpt_names is not None:
        train_ckpts = [ckpt_dir / name for name in train_ckpt_names]
        missing = [p for p in train_ckpts if not p.exists()]
        if missing:
            raise ValueError(f"Missing train checkpoints in {ckpt_dir}: {[p.name for p in missing]}")
    else:
        all_ckpts = sorted(
            ckpt_dir.glob("pinn_ic_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if len(all_ckpts) < N_TRAIN:
            raise ValueError(
                f"Need {N_TRAIN} checkpoints in {ckpt_dir}, found {len(all_ckpts)}."
            )
        train_ckpts = all_ckpts[:N_TRAIN]

    if test_ic_indices_override is not None:
        test_ic_indices = test_ic_indices_override
    else:
        all_ckpts = sorted(
            ckpt_dir.glob("pinn_ic_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if len(all_ckpts) < N_TRAIN + N_TEST:
            raise ValueError(
                f"Need {N_TRAIN + N_TEST} checkpoints in {ckpt_dir}, "
                f"found {len(all_ckpts)}."
            )
        test_ckpts = all_ckpts[N_TRAIN : N_TRAIN + N_TEST]
        test_ic_indices = [int(p.stem.split("_")[-1]) for p in test_ckpts]

    train_ic_indices = [int(p.stem.split("_")[-1]) for p in train_ckpts]

    # Determine time split from first available trajectory
    first_X, _, _ = load_trajectory(dataset, train_ic_indices[0], vae, device)
    n_time  = first_X.shape[-1]
    split_t = int(TRAIN_FRAC * n_time)
    interp_idx = np.arange(0, split_t)
    extrap_idx  = np.arange(split_t, n_time)

    # All time indices (no subsampling — used for both scalar metrics and timeseries)
    ts_idx    = np.arange(n_time)
    time_frac = ts_idx / (n_time - 1)

    # ── Train ICs: one model load per IC ─────────────────────────────────
    train_interp_mses, train_extrap_mses = [], []
    train_mse_curves = []
    n_train_skipped  = 0

    for ckpt_path in tqdm(train_ckpts, desc="  Train ICs", leave=False):
        ic_idx = int(ckpt_path.stem.split("_")[-1])
        X_field, Y_field, z0 = load_trajectory(dataset, ic_idx, vae, device)
        X_flat = X_field.reshape(-1, n_time)
        Y_flat = Y_field.reshape(-1, n_time)
        model  = load_model(ckpt_path, dgpe, n_hidden, device)

        if not is_model_valid(model):
            n_train_skipped += 1
            del model
            continue

        mse_i = evaluate_batched(model, z0, X_flat, Y_flat, interp_idx, STEP, device).mean()
        mse_e = evaluate_batched(model, z0, X_flat, Y_flat, extrap_idx, STEP, device).mean()
        curve = evaluate_batched(model, z0, X_flat, Y_flat, ts_idx,    STEP, device)
        del model

        if not (np.isfinite(mse_i) and np.isfinite(mse_e) and np.isfinite(curve).all()):
            n_train_skipped += 1
            continue

        train_interp_mses.append(float(mse_i))
        train_extrap_mses.append(float(mse_e))
        train_mse_curves.append(curve / curve[0])

    if n_train_skipped:
        print(f"    Skipped {n_train_skipped} diverged train IC(s)")

    # ── Test ICs: cache trajectories, loop models in outer loop ──────────
    _tmp     = load_model(train_ckpts[0], dgpe, n_hidden, device)
    n_params = count_params(_tmp)
    del _tmp

    print(f"    Caching {N_TEST} test trajectories …")
    test_trajs = []
    for ic_idx in test_ic_indices:
        X_field, Y_field, z0 = load_trajectory(dataset, ic_idx, vae, device)
        test_trajs.append((X_field.reshape(-1, n_time), Y_field.reshape(-1, n_time), z0))

    interp_acc    = np.zeros(N_TEST)
    extrap_acc    = np.zeros(N_TEST)
    ts_acc        = np.zeros((N_TEST, n_time))
    valid_counts  = np.zeros(N_TEST, dtype=int)
    n_test_model_skipped = 0

    for ckpt_path in tqdm(train_ckpts, desc="  Test ICs (model loop)", leave=False):
        model = load_model(ckpt_path, dgpe, n_hidden, device)
        if not is_model_valid(model):
            n_test_model_skipped += 1
            del model
            continue
        for j, (X_flat, Y_flat, z0) in enumerate(test_trajs):
            mse_i = evaluate_batched(model, z0, X_flat, Y_flat, interp_idx, STEP, device).mean()
            mse_e = evaluate_batched(model, z0, X_flat, Y_flat, extrap_idx, STEP, device).mean()
            curve = evaluate_batched(model, z0, X_flat, Y_flat, ts_idx,    STEP, device)
            if not (np.isfinite(mse_i) and np.isfinite(mse_e) and np.isfinite(curve).all()):
                continue
            interp_acc[j]   += float(mse_i)
            extrap_acc[j]   += float(mse_e)
            ts_acc[j]       += curve
            valid_counts[j] += 1
        del model

    if n_test_model_skipped:
        print(f"    Skipped {n_test_model_skipped} diverged model(s) during test evaluation")

    # Average over valid models per test IC, then collect per-IC values
    test_interp_mses, test_extrap_mses, test_mse_curves = [], [], []
    for j in range(N_TEST):
        if valid_counts[j] == 0:
            continue
        test_interp_mses.append(interp_acc[j] / valid_counts[j])
        test_extrap_mses.append(extrap_acc[j] / valid_counts[j])
        curve = ts_acc[j] / valid_counts[j]
        test_mse_curves.append(curve / curve[0])

    # Median across ICs (robust to remaining outliers); std for spread
    mse_curve_train = np.median(np.stack(train_mse_curves), axis=0) if train_mse_curves else np.array([])
    mse_curve_test  = np.median(np.stack(test_mse_curves),  axis=0) if test_mse_curves  else np.array([])

    return dict(
        n_hidden=n_hidden,
        istride=istride,
        n_params=n_params,
        n_train_valid=len(train_interp_mses),
        n_test_valid=len(test_interp_mses),
        train_interp_median=float(np.median(train_interp_mses)),
        train_interp_std=float(np.std(train_interp_mses)),
        train_extrap_median=float(np.median(train_extrap_mses)),
        train_extrap_std=float(np.std(train_extrap_mses)),
        test_interp_median=float(np.median(test_interp_mses)),
        test_interp_std=float(np.std(test_interp_mses)),
        test_extrap_median=float(np.median(test_extrap_mses)),
        test_extrap_std=float(np.std(test_extrap_mses)),
        time_frac=time_frac,
        mse_curve_train=mse_curve_train,
        mse_curve_test=mse_curve_test,
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────

TASK_STYLES = [
    # (key_prefix,    label,                         color,     ls,   marker)
    ("train_interp", "Train IC · Interpolation",    "#4C72B0", "-",  "o"),
    ("train_extrap", "Train IC · Extrapolation",    "#4C72B0", "--", "s"),
    ("test_interp",  "Test IC  · Interpolation",    "#DD8452", "-",  "o"),
    ("test_extrap",  "Test IC  · Extrapolation",    "#DD8452", "--", "s"),
]
STAT_KEY = "median"   # central-tendency key suffix used in result dicts


def plot_timeseries(records_sub: list, label_key: str,
                    label_fmt: str, cmap_name: str,
                    title: str, out_path: str,
                    train_frac: float = TRAIN_FRAC):
    """
    Plot MSE vs time fraction for a set of experiments.
    Each config produces two lines: solid = Train IC, dashed = Test IC.

    records_sub : list of result dicts (already filtered to the sweep of interest)
    label_key   : 'n_hidden' or 'istride'
    label_fmt   : format string, e.g. 'n_hidden = {}'
    cmap_name   : matplotlib colormap name for the sweep variable
    """
    from matplotlib.lines import Line2D

    records_sub = sorted(records_sub, key=lambda r: r[label_key])
    cmap = plt.get_cmap(cmap_name, len(records_sub))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, rec in enumerate(records_sub):
        tf    = rec["time_frac"]
        train = rec["mse_curve_train"]
        test  = rec["mse_curve_test"]
        if len(tf) == 0:
            continue
        color = cmap(i)
        label = label_fmt.format(rec[label_key])
        ax.plot(tf, train, color=color, linestyle="-",  linewidth=1.8, label=label)
        ax.plot(tf, test,  color=color, linestyle="--", linewidth=1.8)

    # Vertical line marking interpolation / extrapolation boundary
    ax.axvline(train_frac, color="black", linestyle=":", linewidth=1.4,
               label=f"interp / extrap split ({int(train_frac*100)} %)")

    # Compound legend: color entries (one per config value) + linestyle entries
    handles, labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.8, label="Train IC"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="Test IC"),
    ]
    ax.legend(handles=handles + style_handles, fontsize=9, loc="best",
              ncol=2 if len(records_sub) > 4 else 1)

    ax.set_xlabel("Time  (fraction of trajectory)", fontsize=13)
    ax.set_ylabel("MSE / MSE(t=0)  (averaged over ICs)", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}.{{pdf,png}}")


def plot_scaling(df_sub: pd.DataFrame, x_col: str,
                 xlabel: str, title: str, out_path: str,
                 invert_x: bool = False):
    df_sub = df_sub.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color, ls, marker in TASK_STYLES:
        xs = df_sub[x_col].values
        ys = df_sub[f"{key}_{STAT_KEY}"].values
        es = df_sub[f"{key}_std"].values
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                linewidth=1.8, markersize=7, label=label)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_yscale("log")
    if x_col == "n_params":
        ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}.{{pdf,png}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize PINN IC-generalization scaling results"
    )
    parser.add_argument(
        "--outputs_dir", default="outputs/outputs",
        help="Root directory containing all experiment sub-folders",
    )
    parser.add_argument(
        "--data_dir", default="../Thesis/datasets/gpe_simulations/",
        help="Path to the GPE trajectory dataset",
    )
    parser.add_argument(
        "--vae_checkpoint",
        default="../Thesis/outputs/vae/checkpoints/best_checkpoint.pt",
        help="Path to the pre-trained VAE checkpoint",
    )
    parser.add_argument(
        "--out_dir", default="outputs/scaling_summary",
        help="Where to save the summary CSV and plots",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train_ranking",
        default=None,
        help="Path to all_train_ranking.csv; top N_TRAIN rows per folder are used as train ICs",
    )
    parser.add_argument(
        "--test_ranking",
        default=None,
        help="Path to all_test_ranking.csv; top N_TEST rows per folder are used as test ICs",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load VAE
    print("Loading VAE …")
    vae = VAE(2 * V, latent_dim=LATENT_DIM)
    chkpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(chkpt["model_state_dict"], strict=False)
    vae.to(device).eval()

    # Load dataset
    print("Loading trajectory dataset …")
    dataset = GPEDataset(args.data_dir, normalize=False, mode="trajectories")
    print(f"  {len(dataset)} trajectories available")

    # Build dgpe template (used for neighbour indices only)
    dgpe = build_dgpe()

    # Load per-folder ranking overrides (top N_TRAIN / N_TEST entries per folder)
    train_ranking_by_folder = {}
    if args.train_ranking:
        df_train_rank = pd.read_csv(args.train_ranking)
        for folder, grp in df_train_rank.groupby("folder", sort=False):
            top = grp.head(N_TRAIN)
            train_ranking_by_folder[folder] = top["checkpoint"].tolist()
        print(f"Loaded train ranking: {args.train_ranking}  "
              f"(top {N_TRAIN} per folder, {len(train_ranking_by_folder)} folders)")

    test_ranking_by_folder = {}
    if args.test_ranking:
        df_test_rank = pd.read_csv(args.test_ranking)
        for folder, grp in df_test_rank.groupby("folder", sort=False):
            top = grp.head(N_TEST)
            test_ranking_by_folder[folder] = top["ic_idx"].tolist()
        print(f"Loaded test ranking:  {args.test_ranking}  "
              f"(top {N_TEST} per folder, {len(test_ranking_by_folder)} folders)")

    # Evaluate each experiment
    records = []
    for folder, n_hidden, istride in EXPERIMENTS:
        exp_dir = os.path.join(args.outputs_dir, folder)
        if not os.path.isdir(exp_dir):
            print(f"\n[SKIP] {folder}  (directory not found)")
            continue
        print(f"\n[{folder}]  n_hidden={n_hidden}  istride={istride}")
        train_ckpt_names   = train_ranking_by_folder.get(folder)
        test_ic_idx_list   = test_ranking_by_folder.get(folder)
        if train_ckpt_names:
            print(f"  Using {len(train_ckpt_names)} ranked train checkpoints")
        if test_ic_idx_list:
            print(f"  Using {len(test_ic_idx_list)} ranked test ICs")
        try:
            rec = run_experiment(
                exp_dir, n_hidden, istride, dgpe, vae, dataset, device,
                train_ckpt_names=train_ckpt_names,
                test_ic_indices_override=test_ic_idx_list,
            )
            records.append(rec)
            print(f"  n_params          = {rec['n_params']:,}  "
                  f"(train valid: {rec['n_train_valid']}, test valid: {rec['n_test_valid']})")
            for key, label, *_ in TASK_STYLES:
                print(
                    f"  {label:<36s} "
                    f"{rec[key+'_median']:.4e} ± {rec[key+'_std']:.4e}"
                )
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not records:
        print("\nNo results collected — nothing to plot.")
        return

    import pickle
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.out_dir, "summary_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    pkl_path = os.path.join(args.out_dir, "timeseries_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved: {pkl_path}")

    # ── Plot 1: MSE vs number of parameters (n_hidden sweep) ─────────────
    df_nhid = df[df["istride"] == NHID_SWEEP_ISTRIDE]
    if len(df_nhid) >= 2:
        plot_scaling(
            df_nhid, x_col="n_params",
            xlabel="Number of trainable parameters",
            title=f"Error vs model size  (istride = {NHID_SWEEP_ISTRIDE})",
            out_path=os.path.join(args.out_dir, "plot_vs_n_params"),
        )
    else:
        print(f"  Not enough points for n_params plot (need ≥2, got {len(df_nhid)})")

    # ── Plot 2: MSE vs istride (data density sweep) ───────────────────────
    df_istride = df[df["n_hidden"] == ISTRIDE_SWEEP_NHID]
    if len(df_istride) >= 2:
        plot_scaling(
            df_istride, x_col="istride",
            xlabel="Training time stride  (larger = sparser data)",
            title=f"Error vs data density  (n_hidden = {ISTRIDE_SWEEP_NHID})",
            out_path=os.path.join(args.out_dir, "plot_vs_istride"),
            invert_x=True,   # left = more data = better
        )
    else:
        print(f"  Not enough points for istride plot (need ≥2, got {len(df_istride)})")

    # ── Timeseries plots (MSE vs time, median over all ICs) ──────────────
    recs_nhid    = [r for r in records if r["istride"]  == NHID_SWEEP_ISTRIDE
                    and len(r["time_frac"]) > 0]
    recs_istride = [r for r in records if r["n_hidden"] == ISTRIDE_SWEEP_NHID
                    and len(r["time_frac"]) > 0]

    if len(recs_nhid) >= 2:
        plot_timeseries(
            recs_nhid,
            label_key="n_hidden",
            label_fmt="n_hidden = {}",
            cmap_name="Blues_r",
            title=f"MSE vs time  (istride = {NHID_SWEEP_ISTRIDE},  median over ICs)",
            out_path=os.path.join(args.out_dir, "plot_timeseries_nhid"),
        )
    else:
        print(f"  Not enough points for n_hidden timeseries plot (got {len(recs_nhid)})")

    if len(recs_istride) >= 2:
        plot_timeseries(
            recs_istride,
            label_key="istride",
            label_fmt="istride = {}",
            cmap_name="Greens_r",
            title=f"MSE vs time  (n_hidden = {ISTRIDE_SWEEP_NHID},  median over ICs)",
            out_path=os.path.join(args.out_dir, "plot_timeseries_istride"),
        )
    else:
        print(f"  Not enough points for istride timeseries plot (got {len(recs_istride)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
