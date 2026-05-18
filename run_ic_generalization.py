#!/usr/bin/env python
"""
PINN GPE — Initial-Condition & Time Generalization Study
=========================================================

This script:
1. Loads the pre-trained VAE and the GPE trajectory dataset.
2. Splits trajectories into *train* and *test* initial conditions (ICs).
3. For every train IC, creates a time split:
       - interpolation window  t ∈ [0, T_split)     (first 75 %)
       - extrapolation window  t ∈ [T_split, T_end]  (last  25 %)
   and trains a PINN conditioned on the VAE latent code z₀.
4. After training, evaluates MSE on **all** ICs (train + test) ×
   **both** time windows (interpolation + extrapolation).
5. Produces a 2×2 summary table and a grouped-bar plot.

Usage
-----
    python run_ic_generalization.py [--n_ic_total 10] [--n_ic_train 7] \
                                    [--n_epochs_per_stage 10] [--ne 10] \
                                    [--output_dir outputs/ic_generalization]

All heavy hyper-parameters (loss weights, hidden size, …) are kept
identical to the notebook PINN_GPE_IC.ipynb.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
from src.data.dataset import GPEDataset
from src.dgpe_nn import DGPEModule
from src.models.vae import VAE
from src.dataloaders import generate_datasets
from src.pinn_lib import train_and_validate#_nownb   # headless (no notebook widgets)

from DGPE.GPElib.dynamics_generator import DynamicsGenerator

# ── matplotlib defaults ─────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 14})
sns.set_style("whitegrid")


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_dgpe(beta: float = 1.0, step: float = 0.001, n_steps: int = 5000):
    """Return a DynamicsGenerator pre-configured like the notebook."""
    dgpe = DynamicsGenerator(
        N_part_per_well=1.0,
        W=0,
        disorder_seed=53,
        N_wells=(10, 10, 10),
        dimensionality=3,
        anisotropy=1.0,
        threshold_XY_to_polar=0.25,
        J=1,
        beta=beta,
        integration_method="RK45",
        rtol=1e-8,
        atol=1e-8,
        smooth_quench_to_room=True,
        reset_steps_duration=5,
        calculation_type="lyap_save_all",
        integrator="scipy",
        time=51,
        step=step,
        t_steps=n_steps,
        gamma=1.0,
        quenching_gamma=1.0,
    )
    dgpe.step = step
    dgpe.n_steps = n_steps
    dgpe.icurr = 0
    dgpe.inext = 1
    return dgpe


def encode_ic(vae: VAE, X0: torch.Tensor, Y0: torch.Tensor) -> torch.Tensor:
    """Encode a single initial condition (X0, Y0) → z0  (flat vector)."""
    device = next(vae.parameters()).device
    psi0_flat = torch.stack((X0.flatten(), Y0.flatten())).flatten().to(device)
    mu, logvar = vae.encode(psi0_flat)
    z0 = torch.stack((mu, logvar)).flatten()
    return z0


def prepare_time_arrays(dgpe, X_field, Y_field, step, z0, train_frac=0.75):
    """
    Build (time, psi) arrays and their train / test (interpolation / extrapolation) split.
    The time column is augmented with z0 so each row is [t, z0_1, z0_2, …].
    Returns
    -------
    time_all, psi_all : full arrays  (augmented with z0)
    X_interp, y_interp : interpolation (train) split
    X_extrap, y_extrap : extrapolation (test) split
    X_train, y_train, X_val, y_val : train-val sub-split of the interpolation window
    """
    n_time = X_field.shape[-1]
    time_col = step * np.arange(n_time).reshape(-1, 1)  # (T, 1)
    V = np.prod(X_field.shape[:-1])
    psi_all = np.hstack(
        (
            np.moveaxis(X_field, -1, 0).reshape(n_time, V),
            np.moveaxis(Y_field, -1, 0).reshape(n_time, V),
        )
    )  # (T, 2V)

    # Augment: each row becomes [t, z0_flat]
    z0_np = z0.detach().cpu().numpy().reshape(1, -1)      # (1, latent*2)
    z0_tiled = np.tile(z0_np, (n_time, 1))                # (T, latent*2)
    time_all = np.hstack((time_col, z0_tiled))             # (T, 1+latent*2)

    split_idx = int(train_frac * n_time)
    X_interp = time_all[:split_idx]
    y_interp = psi_all[:split_idx]
    X_extrap = time_all[split_idx:]
    y_extrap = psi_all[split_idx:]

    # sub-split interpolation window into train / val  (75 / 25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_interp, y_interp, test_size=0.25, random_state=0xE2E4
    )
    return (time_all, psi_all,
            X_interp, y_interp,
            X_extrap, y_extrap,
            X_train, y_train, X_val, y_val)


def evaluate_mse(model, vae, z0, X_field, Y_field, time_indices, step):
    """
    Compute MSE of the PINN prediction vs ground truth for the given time indices.

    Parameters
    ----------
    X_field, Y_field : np.ndarray, shape (Nx, Ny, Nz, T)
        Pre-computed trajectory arrays (real / imaginary parts).
    """
    device = next(model.parameters()).device
    mse_total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for idx in time_indices:
            t = torch.tensor([step * idx], dtype=torch.float32, device=device)
            inp = torch.cat((t, z0.detach())).unsqueeze(0)   # (1, 1+latent*2)
            pred = model(inp).squeeze(0)                     # (2V,)
            gt = torch.cat(
                [
                    torch.tensor(X_field[:, :, :, idx].flatten(), dtype=torch.float32, device=device),
                    torch.tensor(Y_field[:, :, :, idx].flatten(), dtype=torch.float32, device=device),
                ]
            )
            mse_total += nn.functional.mse_loss(pred, gt).item()
            count += 1
    return mse_total / max(count, 1)


def evaluate_mse_per_timestep(model, vae, z0, X_field, Y_field, time_indices, step):
    """
    Compute MSE of the PINN prediction vs ground truth at *each* time index.

    Returns
    -------
    times : np.ndarray, shape (len(time_indices),)
    mses  : np.ndarray, shape (len(time_indices),)
    """
    device = next(model.parameters()).device
    times = []
    mses = []
    model.eval()
    with torch.no_grad():
        for idx in time_indices:
            t = torch.tensor([step * idx], dtype=torch.float32, device=device)
            inp = torch.cat((t, z0.detach())).unsqueeze(0)   # (1, 1+latent*2)
            pred = model(inp).squeeze(0)                     # (2V,)
            gt = torch.cat(
                [
                    torch.tensor(X_field[:, :, :, idx].flatten(), dtype=torch.float32, device=device),
                    torch.tensor(Y_field[:, :, :, idx].flatten(), dtype=torch.float32, device=device),
                ]
            )
            times.append(step * idx)
            mses.append(nn.functional.mse_loss(pred, gt).item())
    return np.array(times), np.array(mses)


# ═════════════════════════════════════════════════════════════════════════════
#  Training wrapper  (mirrors the staged schedule from the notebook)
# ═════════════════════════════════════════════════════════════════════════════

def train_pinn_for_ic(
    dgpe, vae, z0,
    X_field, Y_field,
    step, istride,
    ne=10, batch_size=32,
    train_frac=0.75,
    device='cpu',
    n_hidden=128,
):
    """
    Train one PINN model for a single initial condition.
    Returns the trained model.
    """
    (time_all, psi_all,
     X_interp, y_interp,
     X_extrap, y_extrap,
     X_train, y_train, X_val, y_val) = prepare_time_arrays(
        dgpe, X_field, Y_field, step, z0, train_frac=train_frac
    )

    # data-loaders
    train_loader, test_loader, val_loader, init_loader = generate_datasets(
        time_all, psi_all,
        X_train, y_train,
        X_extrap, y_extrap,   # test_loader won't be used during training
        X_val, y_val,
        batch_size, istride,
    )

    # model
    n_in = time_all.shape[-1]                  # 1 + latent_dim*2  (already augmented)
    n_out = psi_all.shape[-1]                  # 2 * V
    model = DGPEModule(dgpe, n_in, n_out, n_hidden=n_hidden).to(device)
    criterion = nn.MSELoss()
    metric = lambda x, y: nn.MSELoss()(x, y)

    # ── staged training schedule (same as notebook) ──────────────────────
    stages = [
        # (lr, epochs_mult, flags)
        (1e-3, 10, dict(criterion_init_cond=True)),
        (1e-3, 10, dict(criterion_init_cond=True, criterion_ibound=True)),
        (1e-3, 10, dict(criterion_init_cond=True, criterion_ibound=True, criterion_Nconst=True)),
        (1e-3, 10, dict(criterion_init_cond=True, criterion_ibound=True, criterion_Nconst=True, criterion_Econst=True)),
        (1e-4, 50, dict(criterion_init_cond=True, criterion_ibound=True, criterion_Nconst=True, criterion_Econst=True, criterion_pinn=True)),
    ]

    for lr, emult, flags in stages:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
        num_epochs = emult * ne
        train_and_validate(
            model, optimizer, criterion, metric,
            train_loader, val_loader, init_loader,
            num_epochs,
            w1=1., w2=1., w3=1., w4=1e-3, w5=1e-6, w6=1., w7=1.,
            device=device,
            verbose=False,
            **flags,
        )

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PINN IC generalization study")
    parser.add_argument("--data_dir", type=str, default="../Thesis/datasets/gpe_simulations/",
                        help="Path to the GPE trajectory dataset")
    parser.add_argument("--vae_checkpoint", type=str,
                        default="../Thesis/outputs/vae/checkpoints/best_checkpoint.pt",
                        help="Path to the pre-trained VAE checkpoint")
    parser.add_argument("--n_ic_total", type=int, default=10,
                        help="Total number of initial conditions to use")
    parser.add_argument("--n_ic_train", type=int, default=7,
                        help="Number of ICs used for training (rest → test)")
    parser.add_argument("--ne", type=int, default=10,
                        help="Epoch multiplier (ne from the notebook)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--istride", type=int, default=100,
                        help="Stride for sub-sampling train time points")
    parser.add_argument("--train_frac", type=float, default=0.75,
                        help="Fraction of time steps for interpolation window")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="GPE interaction parameter β")
    parser.add_argument("--step", type=float, default=0.001,
                        help="GPE integration time step")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Number of GPE integration steps")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="VAE latent dimension")
    parser.add_argument("--V", type=int, default=1000,
                        help="Number of lattice sites (10×10×10)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/ic_generalization",
                        help="Where to save results")
    parser.add_argument("--seed", type=int, default=0xFA1AFE1,
                        help="Random seed")
    parser.add_argument("--n_hidden", type=int, default=128,
                        help="Hidden layer size for the DGPE PINN network")
    parser.add_argument("--load_checkpoints", action="store_true",
                        help="Skip training and load saved model checkpoints from output_dir")
    parser.add_argument("--max_train_ics", type=int, default=None,
                        help="Limit the number of train ICs actually used (for local testing)")
    parser.add_argument("--max_test_ics", type=int, default=None,
                        help="Limit the number of test ICs actually used (for local testing)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda:0', 'cuda:2', or 'auto' (last available GPU)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device selection ────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── 1.  Load VAE ────────────────────────────────────────────────────
    print("Loading VAE …")
    vae = VAE(2 * args.V, latent_dim=args.latent_dim)
    chkpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(chkpt["model_state_dict"], strict=False)
    vae.to(device)
    vae.eval()

    # ── 2.  Load trajectory dataset ─────────────────────────────────────
    print("Loading trajectory dataset …")
    data_traj = GPEDataset(args.data_dir, normalize=False, mode="trajectories")
    n_available = len(data_traj)
    n_ic_total = min(args.n_ic_total, n_available)
    n_ic_train = min(args.n_ic_train, n_ic_total - 1)
    n_ic_test = n_ic_total - n_ic_train

    print(f"  Available trajectories : {n_available}")
    print(f"  Using {n_ic_total} ICs  →  {n_ic_train} train / {n_ic_test} test")

    # pick indices and split
    all_ic_indices = np.random.choice(n_available, size=n_ic_total, replace=False)
    train_ic_idx, test_ic_idx = train_test_split(
        all_ic_indices, train_size=n_ic_train, random_state=42
    )
    # ── Optional truncation for local testing ──────────────────────────
    if args.max_train_ics is not None and len(train_ic_idx) > args.max_train_ics:
        train_ic_idx = train_ic_idx[:args.max_train_ics]
        print(f"  (Truncated train ICs to {args.max_train_ics})")
    if args.max_test_ics is not None and len(test_ic_idx) > args.max_test_ics:
        test_ic_idx = test_ic_idx[:args.max_test_ics]
        print(f"  (Truncated test ICs to {args.max_test_ics})")
    n_ic_train = len(train_ic_idx)
    n_ic_test = len(test_ic_idx)
    all_ic_indices = np.concatenate([train_ic_idx, test_ic_idx])

    print(f"  Train IC indices : {sorted(train_ic_idx)}")
    print(f"  Test  IC indices : {sorted(test_ic_idx)}")

    # ── 3.  Build DynamicsGenerator template (for neighbor indices only) ─
    dgpe = build_dgpe(beta=args.beta, step=args.step, n_steps=args.n_steps)

    # ── 4.  Load pre-computed trajectories from dataset ─────────────────
    print("\nLoading trajectories from dataset …")
    trajectories = {}   # idx → (X_field, Y_field, z0)
    for ic_idx in tqdm(all_ic_indices, desc="Loading ICs"):
        psi_tensor, meta = data_traj[int(ic_idx)]
        # psi_tensor shape: (2, Nx, Ny, Nz, T)
        X_field = psi_tensor[0].numpy()   # (Nx, Ny, Nz, T)
        Y_field = psi_tensor[1].numpy()   # (Nx, Ny, Nz, T)

        X0 = psi_tensor[0, :, :, :, 0]   # (Nx, Ny, Nz)  — initial condition
        Y0 = psi_tensor[1, :, :, :, 0]
        z0 = encode_ic(vae, X0, Y0)

        trajectories[int(ic_idx)] = (X_field, Y_field, z0)

    # ── 5.  Train one PINN per *train* IC  (or load from checkpoints) ──
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    trained_models = {}   # ic_idx → model

    # We need model dimensions to instantiate the architecture
    sample_X_field = next(iter(trajectories.values()))[0]
    sample_z0      = next(iter(trajectories.values()))[2]
    n_time_sample  = sample_X_field.shape[-1]
    V_sample       = int(np.prod(sample_X_field.shape[:-1]))
    n_in  = 1 + sample_z0.shape[0]        # 1 (time) + latent_dim*2
    n_out = 2 * V_sample                  # 2 * V

    if args.load_checkpoints:
        print("\n=== Loading trained PINNs from checkpoints ===")
        skipped = []
        for ic_idx in tqdm(train_ic_idx, desc="Loading models"):
            ic_idx = int(ic_idx)
            ckpt_path = os.path.join(ckpt_dir, f"pinn_ic_{ic_idx}.pt")
            if not os.path.isfile(ckpt_path):
                skipped.append(ic_idx)
                continue
            model = DGPEModule(dgpe, n_in, n_out, n_hidden=args.n_hidden).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            trained_models[ic_idx] = model
            print(f"  Loaded checkpoint for IC {ic_idx}")
        if skipped:
            print(f"  WARNING: {len(skipped)} checkpoint(s) not found (ICs {skipped}), "
                  f"continuing with {len(trained_models)} available model(s).")
        if not trained_models:
            raise FileNotFoundError(
                f"No checkpoints found in {ckpt_dir}. "
                f"Run without --load_checkpoints first to train & save models."
            )
    else:
        print("\n=== Training PINNs (one per train IC) ===")
        for ic_idx in tqdm(train_ic_idx, desc="Training"):
            ic_idx = int(ic_idx)
            X_field, Y_field, z0 = trajectories[ic_idx]
            print(f"\n  Training PINN for IC {ic_idx} …")
            model = train_pinn_for_ic(
                dgpe, vae, z0, X_field, Y_field,
                step=args.step,
                istride=args.istride,
                ne=args.ne,
                batch_size=args.batch_size,
                train_frac=args.train_frac,
                device=device,
                n_hidden=args.n_hidden,
            )
            trained_models[ic_idx] = model
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"pinn_ic_{ic_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    print(f"  {len(trained_models)} models ready.")

    # ── 6.  Evaluate on the 2×2 grid ────────────────────────────────────
    #        rows = IC split  (train / test)
    #        cols = time split (interpolation / extrapolation)
    print("\n=== Evaluating 2×2 grid ===")
    records = []
    # Per-timestep MSE curves, keyed by (ic_split, time_split)
    # Each value is a list of (times_array, mses_array) — one per trajectory
    timestep_curves = {
        ("train", "interpolation"): [],
        ("train", "extrapolation"): [],
        ("test", "interpolation"): [],
        ("test", "extrapolation"): [],
    }

    for ic_idx in tqdm(all_ic_indices, desc="Evaluating"):
        ic_idx = int(ic_idx)
        X_field, Y_field, z0 = trajectories[ic_idx]
        n_time = X_field.shape[-1]
        split_t = int(args.train_frac * n_time)
        interp_indices = np.arange(0, split_t)
        extrap_indices = np.arange(split_t, n_time)

        ic_split = "train" if ic_idx in train_ic_idx else "test"

        # For train ICs, evaluate with their own trained model.
        # For test ICs, evaluate with *every* trained model and report the
        # average (the PINN is conditioned on z0, so we always feed the
        # correct z0 — the question is whether the *shared weights* generalise).
        # Since in the notebook each IC gets its own model, the most meaningful
        # comparison is: pick the single model trained on the closest IC in
        # latent space.

        if ic_split == "train":
            if ic_idx not in trained_models:
                print(f"  Skipping train IC {ic_idx} (no checkpoint/model available)")
                continue
            model = trained_models[ic_idx]
            mse_interp = evaluate_mse(model, vae, z0, X_field, Y_field, interp_indices, args.step)
            mse_extrap = evaluate_mse(model, vae, z0, X_field, Y_field, extrap_indices, args.step)
            records.append(dict(ic_idx=ic_idx, ic_split="train", time_split="interpolation", mse=mse_interp))
            records.append(dict(ic_idx=ic_idx, ic_split="train", time_split="extrapolation", mse=mse_extrap))
            # per-timestep curves
            t_i, m_i = evaluate_mse_per_timestep(model, vae, z0, X_field, Y_field, interp_indices, args.step)
            t_e, m_e = evaluate_mse_per_timestep(model, vae, z0, X_field, Y_field, extrap_indices, args.step)
            timestep_curves[("train", "interpolation")].append((t_i, m_i))
            timestep_curves[("train", "extrapolation")].append((t_e, m_e))
        else:
            # Test IC: evaluate with every trained model (different weights),
            # but always condition on the correct z0 of this test IC.
            for train_idx, model in trained_models.items():
                mse_interp = evaluate_mse(model, vae, z0, X_field, Y_field, interp_indices, args.step)
                mse_extrap = evaluate_mse(model, vae, z0, X_field, Y_field, extrap_indices, args.step)
                records.append(dict(ic_idx=ic_idx, ic_split="test", time_split="interpolation",
                                    mse=mse_interp, trained_on_ic=train_idx))
                records.append(dict(ic_idx=ic_idx, ic_split="test", time_split="extrapolation",
                                    mse=mse_extrap, trained_on_ic=train_idx))
                # per-timestep curves
                t_i, m_i = evaluate_mse_per_timestep(model, vae, z0, X_field, Y_field, interp_indices, args.step)
                t_e, m_e = evaluate_mse_per_timestep(model, vae, z0, X_field, Y_field, extrap_indices, args.step)
                timestep_curves[("test", "interpolation")].append((t_i, m_i))
                timestep_curves[("test", "extrapolation")].append((t_e, m_e))

    df = pd.DataFrame(records)

    # ── 7.  Aggregate into the 2×2 table ────────────────────────────────
    # For test ICs evaluated with multiple models, take the mean over models
    # (you could also take the best or worst).
    table = (
        df.groupby(["ic_split", "time_split"])["mse"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # pivot for a nice 2×2 view
    pivot_mean = table.pivot(index="ic_split", columns="time_split", values="mean")
    pivot_std = table.pivot(index="ic_split", columns="time_split", values="std")

    print("\n" + "=" * 60)
    print("  2×2 MSE TABLE  (mean ± std)")
    print("=" * 60)
    display_table = pivot_mean.copy()
    for col in display_table.columns:
        display_table[col] = [
            f"{pivot_mean.loc[r, col]:.4e} ± {pivot_std.loc[r, col]:.4e}"
            for r in display_table.index
        ]
    print(display_table.to_string())
    print("=" * 60)

    # save raw results
    csv_path = os.path.join(args.output_dir, "results_raw.csv")
    df.to_csv(csv_path, index=False)
    table_path = os.path.join(args.output_dir, "results_2x2.csv")
    table.to_csv(table_path, index=False)
    print(f"\nRaw results → {csv_path}")
    print(f"2×2 table   → {table_path}")

    # ── 8.  Plotting ────────────────────────────────────────────────────
    # 8a.  Grouped bar chart  (2×2)
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_df = df.groupby(["ic_split", "time_split"])["mse"].mean().reset_index()
    bar_df["label"] = bar_df["ic_split"] + "\n" + bar_df["time_split"]
    colors = {"train": "#4C72B0", "test": "#DD8452"}
    hatches = {"interpolation": "", "extrapolation": "//"}

    x_pos = np.arange(len(bar_df))
    bars = ax.bar(x_pos, bar_df["mse"],
                  color=[colors[r] for r in bar_df["ic_split"]],
                  edgecolor="black", linewidth=0.8)
    for bar, ts in zip(bars, bar_df["time_split"]):
        bar.set_hatch(hatches[ts])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_df["label"], fontsize=11)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_title("PINN prediction error:\nIC split × Time split", fontsize=14)
    ax.set_yscale("log")
    # custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["train"], label="Train IC"),
        Patch(facecolor=colors["test"], label="Test IC"),
        Patch(facecolor="white", edgecolor="black", label="Interpolation"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Extrapolation"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "bar_2x2.pdf"), dpi=150)
    fig.savefig(os.path.join(args.output_dir, "bar_2x2.png"), dpi=150)
    print(f"Bar plot    → {args.output_dir}/bar_2x2.{{pdf,png}}")

    # 8b.  Per-IC scatter (MSE vs IC index, coloured by split)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax2, ts in zip(axes2, ["interpolation", "extrapolation"]):
        sub = df[df["time_split"] == ts].groupby(["ic_idx", "ic_split"])["mse"].mean().reset_index()
        for ic_sp, marker, c in [("train", "o", colors["train"]), ("test", "s", colors["test"])]:
            part = sub[sub["ic_split"] == ic_sp]
            ax2.scatter(part["ic_idx"], part["mse"], c=c, marker=marker, s=80,
                        edgecolors="k", linewidths=0.5, label=f"{ic_sp} IC", zorder=3)
        ax2.set_xlabel("IC index", fontsize=12)
        ax2.set_title(ts.capitalize(), fontsize=13)
        ax2.set_yscale("log")
        ax2.legend(fontsize=10)
    axes2[0].set_ylabel("MSE", fontsize=12)
    fig2.suptitle("Per-IC prediction error", fontsize=14, y=1.02)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "scatter_per_ic.pdf"), dpi=150)
    fig2.savefig(os.path.join(args.output_dir, "scatter_per_ic.png"), dpi=150)
    print(f"Scatter     → {args.output_dir}/scatter_per_ic.{{pdf,png}}")

    # 8c.  Heatmap of the 2×2 table
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_mean, annot=True, fmt=".3e", cmap="YlOrRd",
                linewidths=1, ax=ax3, cbar_kws={"label": "MSE"})
    ax3.set_title("Mean MSE — 2×2 Grid", fontsize=14)
    ax3.set_ylabel("IC split")
    ax3.set_xlabel("Time split")
    plt.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "heatmap_2x2.pdf"), dpi=150)
    fig3.savefig(os.path.join(args.output_dir, "heatmap_2x2.png"), dpi=150)
    print(f"Heatmap     → {args.output_dir}/heatmap_2x2.{{pdf,png}}")

    # 8d.  Box plot (richer view of the distribution)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="time_split", y="mse", hue="ic_split",
                palette=colors, ax=ax4)
    ax4.set_yscale("log")
    ax4.set_ylabel("MSE", fontsize=12)
    ax4.set_xlabel("Time split", fontsize=12)
    ax4.set_title("MSE distribution by IC & time split", fontsize=14)
    plt.tight_layout()
    fig4.savefig(os.path.join(args.output_dir, "boxplot_2x2.pdf"), dpi=150)
    fig4.savefig(os.path.join(args.output_dir, "boxplot_2x2.png"), dpi=150)
    print(f"Boxplot     → {args.output_dir}/boxplot_2x2.{{pdf,png}}")

    # 8e.  MSE vs time  (averaged over trajectories, one curve per case)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    case_styles = {
        ("train", "interpolation"): dict(color="#4C72B0", linestyle="-",  label="Train IC / Interpolation"),
        ("train", "extrapolation"): dict(color="#4C72B0", linestyle="--", label="Train IC / Extrapolation"),
        ("test",  "interpolation"): dict(color="#DD8452", linestyle="-",  label="Test IC / Interpolation"),
        ("test",  "extrapolation"): dict(color="#DD8452", linestyle="--", label="Test IC / Extrapolation"),
    }
    for key, curves in timestep_curves.items():
        if not curves:
            continue
        # All curves in the same case share the same time grid,
        # so we can stack and average.
        times_ref = curves[0][0]
        mse_stack = np.stack([m for _, m in curves], axis=0)   # (n_curves, n_times)
        mse_mean = mse_stack.mean(axis=0)
        mse_std  = mse_stack.std(axis=0)
        style = case_styles[key]
        ax5.plot(times_ref, mse_mean, linewidth=2, **style)
        ax5.fill_between(times_ref, mse_mean - mse_std, mse_mean + mse_std,
                         alpha=0.15, color=style["color"])

    # Draw a vertical line at the interpolation/extrapolation boundary
    sample_X = next(iter(trajectories.values()))[0]
    n_time_sample = sample_X.shape[-1]
    t_split_val = args.step * int(args.train_frac * n_time_sample)
    ax5.axvline(t_split_val, color="grey", linestyle=":", linewidth=1.5,
                label=f"Time split (t={t_split_val:.2f})")

    ax5.set_xlabel("Time", fontsize=13)
    ax5.set_ylabel("MSE", fontsize=13)
    ax5.set_title("MSE vs Time  (averaged over trajectories)", fontsize=14)
    ax5.set_yscale("log")
    ax5.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    fig5.savefig(os.path.join(args.output_dir, "mse_vs_time.pdf"), dpi=150)
    fig5.savefig(os.path.join(args.output_dir, "mse_vs_time.png"), dpi=150)
    print(f"MSE vs time → {args.output_dir}/mse_vs_time.{{pdf,png}}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
