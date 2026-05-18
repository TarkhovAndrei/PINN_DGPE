#!/usr/bin/env python
"""
Rank trained PINN models by prediction error.

For each experiment folder two rankings are produced:

  Train ranking — each model evaluated on its own IC at the actual training
                  time points (istride-spaced within the interpolation window,
                  i.e. the points the model explicitly saw during training).

  Test ranking  — each model evaluated on all N_TEST held-out ICs at every
                  time point; the per-IC MSEs are aggregated as the median
                  across test ICs.  Diverged models are skipped.

Outputs (one file per experiment, per ranking):
  <out_dir>/<folder>_train_ranking.txt
  <out_dir>/<folder>_test_ranking.txt
  <out_dir>/all_train_ranking.csv   (combined, all experiments)
  <out_dir>/all_test_ranking.csv

Usage:
    python rank_models.py \
        --outputs_dir outputs/outputs \
        --data_dir    ../Thesis/datasets/gpe_simulations/ \
        --vae_checkpoint ../Thesis/outputs/vae/checkpoints/best_checkpoint.pt \
        --out_dir outputs/rankings
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

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

# ── Experiment registry (folder, n_hidden, istride) ──────────────────────────
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

LATENT_DIM = 128
V          = 1000
N_IN       = 1 + LATENT_DIM * 2
N_OUT      = 2 * V
STEP       = 0.001
N_STEPS    = 5000
TRAIN_FRAC = 0.75

N_TRAIN = 50
N_TEST  = 50


# ── Helpers (mirrors summarize_scaling.py) ────────────────────────────────────

def build_dgpe():
    dgpe = DynamicsGenerator(
        N_part_per_well=1.0, W=0, disorder_seed=53,
        N_wells=(10, 10, 10), dimensionality=3, anisotropy=1.0,
        threshold_XY_to_polar=0.25, J=1, beta=1.0,
        integration_method="RK45", rtol=1e-8, atol=1e-8,
        smooth_quench_to_room=True, reset_steps_duration=5,
        calculation_type="lyap_save_all", integrator="scipy",
        time=51, step=STEP, t_steps=N_STEPS, gamma=1.0, quenching_gamma=1.0,
    )
    dgpe.step = STEP;  dgpe.n_steps = N_STEPS
    dgpe.icurr = 0;    dgpe.inext = 1
    return dgpe


def encode_ic(vae, X0, Y0, device):
    psi0 = torch.stack((X0.flatten(), Y0.flatten())).flatten().to(device)
    mu, logvar = vae.encode(psi0)
    return torch.stack((mu, logvar)).flatten()


def load_trajectory(dataset, ic_idx, vae, device):
    psi, _ = dataset[int(ic_idx)]
    X_field = psi[0].numpy()
    Y_field = psi[1].numpy()
    z0 = encode_ic(vae, psi[0, :, :, :, 0], psi[1, :, :, :, 0], device)
    return X_field, Y_field, z0


def load_model(ckpt_path, dgpe, n_hidden, device):
    model = DGPEModule(dgpe, N_IN, N_OUT, n_hidden=n_hidden).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    return model


def is_model_valid(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def evaluate_batched(model, z0, X_flat, Y_flat, time_indices, device):
    """Single batched forward pass; returns per-timestep MSE array."""
    n = len(time_indices)
    t_batch = torch.tensor(STEP * time_indices, dtype=torch.float32, device=device).unsqueeze(1)
    z0_exp  = z0.detach().unsqueeze(0).expand(n, -1)
    inp     = torch.cat([t_batch, z0_exp], dim=1)
    gt = torch.tensor(
        np.concatenate([X_flat[:, time_indices].T, Y_flat[:, time_indices].T], axis=1),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        pred = model(inp)
        per_ts = ((pred - gt) ** 2).mean(dim=1).cpu().numpy()
    return per_ts


# ── Per-experiment ranking ────────────────────────────────────────────────────

def rank_experiment(exp_dir, n_hidden, istride, dgpe, vae, dataset, device):
    """
    Returns two sorted lists of dicts:
      train_rows : ranked by MSE on own IC at training time points (istride-spaced)
      test_rows  : ranked by median MSE across N_TEST held-out ICs (all time points)

    Each dict has keys: ic_idx, checkpoint, mse, rank, n_hidden, istride, folder.
    """
    ckpt_dir = Path(exp_dir) / "checkpoints"
    all_ckpts = sorted(
        ckpt_dir.glob("pinn_ic_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if len(all_ckpts) < N_TRAIN + N_TEST:
        raise ValueError(
            f"Need {N_TRAIN + N_TEST} checkpoints in {ckpt_dir}, "
            f"found {len(all_ckpts)}."
        )

    train_ckpts = all_ckpts[:N_TRAIN]
    test_ckpts  = all_ckpts[N_TRAIN : N_TRAIN + N_TEST]
    test_ic_indices = [int(p.stem.split("_")[-1]) for p in test_ckpts]

    # Time index arrays
    n_time   = N_STEPS
    split_t  = int(TRAIN_FRAC * n_time)
    # Training points: istride-spaced within the interpolation window
    train_t_idx = np.arange(0, split_t, istride)
    # All time points (for test ranking)
    all_t_idx   = np.arange(n_time)

    folder_name = Path(exp_dir).name

    # ── Train ranking ─────────────────────────────────────────────────────────
    train_rows = []
    for ckpt_path in tqdm(train_ckpts, desc="  Train ranking", leave=False):
        ic_idx = int(ckpt_path.stem.split("_")[-1])
        X_field, Y_field, z0 = load_trajectory(dataset, ic_idx, vae, device)
        X_flat = X_field.reshape(-1, n_time)
        Y_flat = Y_field.reshape(-1, n_time)
        model  = load_model(ckpt_path, dgpe, n_hidden, device)

        if not is_model_valid(model):
            del model
            continue

        per_ts = evaluate_batched(model, z0, X_flat, Y_flat, train_t_idx, device)
        del model

        if not np.isfinite(per_ts).all():
            continue

        train_rows.append(dict(
            folder=folder_name, n_hidden=n_hidden, istride=istride,
            ic_idx=ic_idx, checkpoint=ckpt_path.name,
            mse=float(per_ts.mean()),
            n_train_pts=len(train_t_idx),
        ))

    train_rows.sort(key=lambda r: r["mse"])
    for rank, row in enumerate(train_rows, start=1):
        row["rank"] = rank

    # ── Test ranking — cache test trajectories, loop models ───────────────────
    print(f"    Caching {len(test_ic_indices)} test trajectories …")
    test_trajs = []
    for ic_idx in test_ic_indices:
        X_field, Y_field, z0 = load_trajectory(dataset, ic_idx, vae, device)
        test_trajs.append((ic_idx, X_field.reshape(-1, n_time), Y_field.reshape(-1, n_time), z0))

    # Accumulate per-test-IC MSE for each train model
    # Shape: (N_TRAIN, N_TEST) — rows = train models, cols = test ICs
    # We use a list of lists to handle skipped (diverged) models cleanly.
    model_test_mses = {}   # ckpt_name → array of per-test-IC MSE

    for ckpt_path in tqdm(train_ckpts, desc="  Test ranking (model loop)", leave=False):
        model = load_model(ckpt_path, dgpe, n_hidden, device)
        if not is_model_valid(model):
            del model
            continue

        per_ic_mse = []
        for _, X_flat, Y_flat, z0 in test_trajs:
            per_ts = evaluate_batched(model, z0, X_flat, Y_flat, all_t_idx, device)
            per_ic_mse.append(float(per_ts.mean()) if np.isfinite(per_ts).all() else np.nan)
        del model

        arr = np.array(per_ic_mse)
        if np.isfinite(arr).any():
            model_test_mses[ckpt_path.name] = arr

    test_rows = []
    for ckpt_path in train_ckpts:
        name = ckpt_path.name
        if name not in model_test_mses:
            continue   # diverged model
        arr    = model_test_mses[name]
        ic_idx = int(ckpt_path.stem.split("_")[-1])
        # Median over test ICs (ignoring NaN from any diverged predictions)
        median_mse = float(np.nanmedian(arr))
        if not np.isfinite(median_mse):
            continue
        test_rows.append(dict(
            folder=folder_name, n_hidden=n_hidden, istride=istride,
            ic_idx=ic_idx, checkpoint=name,
            median_mse=median_mse,
            std_mse=float(np.nanstd(arr)),
            n_test_ics=int(np.isfinite(arr).sum()),
        ))

    test_rows.sort(key=lambda r: r["median_mse"])
    for rank, row in enumerate(test_rows, start=1):
        row["rank"] = rank

    return train_rows, test_rows


# ── Text file writer ──────────────────────────────────────────────────────────

def write_train_ranking(path, rows, n_hidden, istride, folder):
    split_t = int(TRAIN_FRAC * N_STEPS)
    n_pts   = len(np.arange(0, split_t, istride))
    with open(path, "w") as f:
        f.write(f"Train-set ranking — {folder}\n")
        f.write(f"n_hidden={n_hidden}  istride={istride}\n")
        f.write(f"Evaluated on {n_pts} training points "
                f"(every {istride}-th step in t ∈ [0, {TRAIN_FRAC*100:.0f}%) of trajectory)\n")
        f.write(f"Valid models ranked: {len(rows)}\n")
        f.write("\n")
        f.write(f"{'Rank':>5}  {'IC idx':>8}  {'Checkpoint':<30}  {'Train MSE':>14}  {'# train pts':>11}\n")
        f.write("-" * 80 + "\n")
        for r in rows:
            f.write(
                f"{r['rank']:>5}  {r['ic_idx']:>8}  {r['checkpoint']:<30}  "
                f"{r['mse']:>14.6e}  {r['n_train_pts']:>11}\n"
            )
    print(f"  Saved {path}")


def write_test_ranking(path, rows, n_hidden, istride, folder):
    with open(path, "w") as f:
        f.write(f"Test-set ranking — {folder}\n")
        f.write(f"n_hidden={n_hidden}  istride={istride}\n")
        f.write(f"Evaluated on all {N_STEPS} time points across {N_TEST} held-out ICs\n")
        f.write(f"Central tendency: median MSE across test ICs\n")
        f.write(f"Valid models ranked: {len(rows)}\n")
        f.write("\n")
        f.write(
            f"{'Rank':>5}  {'IC idx':>8}  {'Checkpoint':<30}  "
            f"{'Median test MSE':>16}  {'Std test MSE':>13}  {'# valid ICs':>11}\n"
        )
        f.write("-" * 95 + "\n")
        for r in rows:
            f.write(
                f"{r['rank']:>5}  {r['ic_idx']:>8}  {r['checkpoint']:<30}  "
                f"{r['median_mse']:>16.6e}  {r['std_mse']:>13.6e}  {r['n_test_ics']:>11}\n"
            )
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank PINN models by prediction error")
    parser.add_argument("--outputs_dir",    default="outputs/outputs")
    parser.add_argument("--data_dir",       default="../Thesis/datasets/gpe_simulations/")
    parser.add_argument("--vae_checkpoint", default="../Thesis/outputs/vae/checkpoints/best_checkpoint.pt")
    parser.add_argument("--out_dir",        default="outputs/rankings")
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(f"cuda:{torch.cuda.device_count()-1}") \
                 if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading VAE …")
    vae = VAE(2 * V, latent_dim=LATENT_DIM)
    chkpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(chkpt["model_state_dict"], strict=False)
    vae.to(device).eval()

    print("Loading dataset …")
    dataset = GPEDataset(args.data_dir, normalize=False, mode="trajectories")
    print(f"  {len(dataset)} trajectories available")

    dgpe = build_dgpe()

    all_train_rows = []
    all_test_rows  = []

    for folder, n_hidden, istride in EXPERIMENTS:
        exp_dir = os.path.join(args.outputs_dir, folder)
        if not os.path.isdir(exp_dir):
            print(f"\n[SKIP] {folder}  (not found)")
            continue

        print(f"\n[{folder}]  n_hidden={n_hidden}  istride={istride}")
        try:
            train_rows, test_rows = rank_experiment(
                exp_dir, n_hidden, istride, dgpe, vae, dataset, device
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        # Per-experiment text files
        base = os.path.join(args.out_dir, folder)
        write_train_ranking(f"{base}_train_ranking.txt", train_rows, n_hidden, istride, folder)
        write_test_ranking( f"{base}_test_ranking.txt",  test_rows,  n_hidden, istride, folder)

        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)

        if train_rows:
            best = train_rows[0]
            print(f"  Train best : IC {best['ic_idx']:>4}  MSE = {best['mse']:.4e}")
        if test_rows:
            best = test_rows[0]
            print(f"  Test  best : IC {best['ic_idx']:>4}  median MSE = {best['median_mse']:.4e}")

    # Combined CSV across all experiments
    if all_train_rows:
        df_train = pd.DataFrame(all_train_rows)
        df_train.sort_values("mse", inplace=True)
        df_train.insert(0, "global_rank", range(1, len(df_train) + 1))
        csv_path = os.path.join(args.out_dir, "all_train_ranking.csv")
        df_train.to_csv(csv_path, index=False)
        print(f"\nSaved combined train ranking → {csv_path}")

    if all_test_rows:
        df_test = pd.DataFrame(all_test_rows)
        df_test.sort_values("median_mse", inplace=True)
        df_test.insert(0, "global_rank", range(1, len(df_test) + 1))
        csv_path = os.path.join(args.out_dir, "all_test_ranking.csv")
        df_test.to_csv(csv_path, index=False)
        print(f"Saved combined test  ranking → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
