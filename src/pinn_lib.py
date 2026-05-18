import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import clear_output
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
from torch.func import jvp
import wandb
import os

def plot_losses(train_losses, train_metrics, val_losses, val_metrics, axs, label='None'):
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label=f"{label}_train")
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label=f"{label}_val")
    axs[1].plot(range(1, len(train_metrics) + 1), train_metrics, label=f"{label}_train")
    axs[1].plot(range(1, len(val_metrics) + 1), val_metrics, label=f"{label}_val")

    if max(train_losses) / min(train_losses) > 10:
        axs[0].set_yscale("log")

    if max(train_metrics) / min(train_metrics) > 10:
        axs[0].set_yscale("log")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)

    axs[0].set_ylabel("loss")
    axs[1].set_ylabel("MSE")
    # plt.show()

def train_and_validate(
    model,
    optimizer,
    criterion,
    metric,
    train_loader,
    val_loader,
    init_loader,
    num_epochs,
    vae=None,
    criterion_init_cond=False,
    criterion_ibound=False,
    criterion_pinn=False,
    criterion_Nconst=False,
    criterion_Econst=False,
    criterion_pinn_model=False,
    criterion_periodic=False,
    verbose=True,
    w1=1,
    w2=1,
    w3=1,
    w4=1,
    w5=1,
    w6=1,
    w7=1,
    device='cpu',
    run=None,
    patience=None,
    etol=1e-4,
    max_grad_norm=100.0
):
    """
    Train and validate neural network
      - model: neural network to train
      - optimizer: optimizer chained to a model
      - criterion: loss function class
      - metric: function to measure MSE taking neural networks predictions
                 and ground truth labels
      - train_loader: DataLoader with train set
      - val_loader: DataLoader with validation set
      - num_epochs: number of epochs to train
      - verbose: whether to plot metrics during training
    Returns:
      - train_mse: training MSE over the last epoch
      - val_mse: validation MSE after the last epoch
    """

    _t0_init, _psi0_init = next(iter(init_loader))
    init_loader = DataLoader(
        TensorDataset(_t0_init.to(device), _psi0_init.to(device)),
        batch_size=_t0_init.shape[0],
    )

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    train_losses1, val_losses1 = [], []
    train_metrics1, val_metrics1 = [], []
    train_losses2, val_losses2 = [], []
    train_metrics2, val_metrics2 = [], []
    train_losses3, val_losses3 = [], []
    train_metrics3, val_metrics3 = [], []
    train_losses4, val_losses4 = [], []
    train_metrics4, val_metrics4 = [], []
    train_losses5, val_losses5 = [], []
    train_metrics5, val_metrics5 = [], []
    train_losses6, val_losses6 = [], []
    train_metrics6, val_metrics6 = [], []
    train_losses7, val_losses7 = [], []
    train_metrics7, val_metrics7 = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        running_loss1, running_metric1 = 0, 0
        running_loss2, running_metric2 = 0, 0
        running_loss3, running_metric3 = 0, 0
        running_loss4, running_metric4 = 0, 0
        running_loss5, running_metric5 = 0, 0
        running_loss6, running_metric6 = 0, 0
        running_loss7, running_metric7 = 0, 0
        pbar = (
            tqdm(train_loader, desc=f"Training {epoch}/{num_epochs}")
            if verbose
            else train_loader
        )
        interm_losses = []
        interm_metrics = []

        pre_epoch_state = {k: v.clone() for k, v in model.state_dict().items()}
        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            def closure():
                t0, psi0 = next(iter(init_loader))
                if vae is not None:
                    z0 = torch.stack(vae.encode(psi0)).flatten()        
                    X_batch_aug = torch.cat((X_batch, z0.unsqueeze(0).repeat(X_batch.shape[0],1)), dim=1)
                else:
                    X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    # print(model(t0).shape, psi0.shape)
                    if vae is not None:
                        loss1 =  w1 * criterion(model(torch.cat((t0, z0.unsqueeze(0)), dim=1)), psi0)
                    else:
                        loss1 =  w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-loss)])).requires_grad_(False)
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * loss1), torch.exp(-w2 * loss2)])).requires_grad_(False)
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = w7 * criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += loss7
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent explosion
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # If loss is NaN/Inf, zero out grads so LBFGS skips the update
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                return loss
            optimizer.step(closure)
            # optimizer.step()

            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                if verbose:
                    print(f"NaN detected in weights at epoch {epoch}, batch {i}. Rolling back to pre-epoch checkpoint.")
                model.load_state_dict(pre_epoch_state)
                break  # skip the rest of this epoch

            with torch.no_grad():
                t0, psi0 = next(iter(init_loader))
                if vae is not None:
                    z0 = torch.stack(vae.encode(psi0)).flatten()        
                    X_batch_aug = torch.cat((X_batch, z0.unsqueeze(0).repeat(X_batch.shape[0],1)), dim=1)
                else:
                    X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    # print(model(t0).shape, psi0.shape)
                    if vae is not None:
                        z0 = torch.stack(vae.encode(psi0)).flatten()        
                        loss1 =  w1 * criterion(model(torch.cat((t0, z0.unsqueeze(0)), dim=1)), psi0)
                    else:
                        loss1 = w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-loss)])).requires_grad_(False)
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * loss1), torch.exp(-w2 * loss2)])).requires_grad_(False)
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = w7 * criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += loss7
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    if vae is not None:
                        z0 = torch.stack(vae.encode(psi0)).flatten()        
                        metric_value1 = w1 * metric(model(torch.cat((t0, z0.unsqueeze(0)), dim=1)), psi0)
                    else:
                        metric_value1 = w1 * metric(model(t0), psi0)
                    metric_value = metric_value1
                    metric_value2, metric_value3, metric_value4, metric_value5, metric_value6, metric_value7 = metric_value1, metric_value1, metric_value1, metric_value1, metric_value1, metric_value1
    
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-metric_value1)])).requires_grad_(False)
                if criterion_ibound:
                    metric_value2 = w2 * metric(predictions, y_batch)
                    metric_value += metric_value2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * metric_value1), torch.exp(-w2 * metric_value2)])).requires_grad_(False)
                if criterion_pinn:
                    metric_value3 = w3 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch))
                    metric_value += metric_value3
                if criterion_Nconst:
                    metric_value4 = w4 * metric(model.N(model(X_batch_aug)), model.N(y_batch))
                    metric_value += metric_value4
                if criterion_Econst:
                    metric_value5 = w5 * metric(model.E(model(X_batch_aug)), model.E(y_batch))
                    metric_value += metric_value5
                if criterion_pinn_model:
                    metric_value6 = w6 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    metric_value += metric_value6
                if criterion_periodic:
                    metric_value7 = w7 *metric(predictions, predictions[:,model.nn_idx_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idx_2_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idy_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idy_2_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idz_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idz_2_full])
                    metric_value +=  metric_value7
            
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]
                interm_losses += [1. * running_loss / (i+1)]
                interm_metrics += [1. * running_metric / (i+1)]
                # print(interm_losses)
                running_loss1 += loss1.item() * X_batch.shape[0]
                running_metric1 += metric_value1 * X_batch.shape[0]

                running_loss2 += loss2.item() * X_batch.shape[0]
                running_metric2 += metric_value2 * X_batch.shape[0]

                running_loss3 += loss3.item() * X_batch.shape[0]
                running_metric3 += metric_value3 * X_batch.shape[0]

                running_loss4 += loss4.item() * X_batch.shape[0]
                running_metric4 += metric_value4 * X_batch.shape[0]
                
                running_loss5 += loss5.item() * X_batch.shape[0]
                running_metric5 += metric_value5 * X_batch.shape[0]

                running_loss6 += loss6.item() * X_batch.shape[0]
                running_metric6 += metric_value6 * X_batch.shape[0]

                running_loss7 += loss7.item() * X_batch.shape[0]
                running_metric7 += metric_value7 * X_batch.shape[0]


        train_losses += [running_loss / len(train_loader.dataset)]
        train_metrics += [running_metric / len(train_loader.dataset)]
        train_losses1 += [running_loss1 / len(train_loader.dataset)]
        train_metrics1 += [running_metric1 / len(train_loader.dataset)]
        train_losses2 += [running_loss2 / len(train_loader.dataset)]
        train_metrics2 += [running_metric2 / len(train_loader.dataset)]
        train_losses3 += [running_loss3 / len(train_loader.dataset)]
        train_metrics3 += [running_metric3 / len(train_loader.dataset)]
        train_losses4 += [running_loss4 / len(train_loader.dataset)]
        train_metrics4 += [running_metric4 / len(train_loader.dataset)]
        train_losses5 += [running_loss5 / len(train_loader.dataset)]
        train_metrics5 += [running_metric5 / len(train_loader.dataset)]
        train_losses6 += [running_loss6 / len(train_loader.dataset)]
        train_metrics6 += [running_metric6 / len(train_loader.dataset)]
        train_losses7 += [running_loss7 / len(train_loader.dataset)]
        train_metrics7 += [running_metric7 / len(train_loader.dataset)]
        if run is not None:
            run.log(
                    # pbar.set_postfix(
                        {"train_loss": train_losses[-1], "train_MSE": train_metrics[-1],
                                     "train_loss1_init": train_losses1[-1], "train_MSE1_init": train_metrics1[-1],
                                      "train_loss2_sampling": train_losses2[-1], "train_MSE2_sampling": train_metrics2[-1],
                                      "train_loss3_pinn": train_losses3[-1], "train_MSE3_pinn": train_metrics3[-1],
                                      "train_loss4_n": train_losses4[-1], "train_MSE4_n": train_metrics4[-1],
                                      "train_loss5_e": train_losses5[-1], "train_MSE5_e": train_metrics5[-1],
                                      "train_loss6_pinn_model": train_losses6[-1], "train_MSE6_pinn_model": train_metrics6[-1],
                                      "train_loss7_boundary_periodic": train_losses7[-1], "train_MSE7_boundary_periodic": train_metrics7[-1],
                                     })#, step=epoch)
        model.eval()
        running_loss, running_metric = 0, 0
        running_loss1, running_metric1 = 0, 0
        running_loss2, running_metric2 = 0, 0
        running_loss3, running_metric3 = 0, 0
        running_loss4, running_metric4 = 0, 0
        running_loss5, running_metric5 = 0, 0
        running_loss6, running_metric6 = 0, 0
        running_loss7, running_metric7 = 0, 0
        pbar = (
            tqdm(val_loader, desc=f"Validating {epoch}/{num_epochs}")
            if verbose
            else val_loader
        )

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                t0, psi0 = next(iter(init_loader))
                if vae is not None:
                    z0 = torch.stack(vae.encode(psi0)).flatten()        
                    X_batch_aug = torch.cat((X_batch, z0.unsqueeze(0).repeat(X_batch.shape[0],1)), dim=1)
                else:
                    X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    if vae is not None:
                        loss1 =  w1 * criterion(model(torch.cat((t0, z0.unsqueeze(0)), dim=1)), psi0)
                    else:
                        loss1 =  w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    #normalize by t=0
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-w1*loss1)]))
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1*loss1), torch.exp(-w2 * loss2)]))
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += w7 * loss7
                
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    if vae is not None:
                        metric_value1 = w1 * metric(model(torch.cat((t0, z0.unsqueeze(0)), dim=1)), psi0)
                    else:
                        metric_value1 = w1 * metric(model(t0), psi0)
                    metric_value = metric_value1
                    metric_value2, metric_value3, metric_value4, metric_value5, metric_value6, metric_value7 = metric_value1, metric_value1, metric_value1, metric_value1, metric_value1, metric_value1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-metric_value1)])).requires_grad_(False)
                if criterion_ibound:
                    metric_value2 = w2 * metric(predictions, y_batch)
                    metric_value += metric_value2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * metric_value1), torch.exp(-w2 * metric_value2)])).requires_grad_(False)
                if criterion_pinn:
                    metric_value3 = w3 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch))
                    metric_value += metric_value3
                if criterion_Nconst:
                    metric_value4 = w4 * metric(model.N(model(X_batch_aug)), model.N(y_batch))
                    metric_value += metric_value4
                if criterion_Econst:
                    metric_value5 = w5 * metric(model.E(model(X_batch_aug)), model.E(y_batch))
                    metric_value += metric_value5
                if criterion_pinn_model:
                    metric_value6 = w6 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    metric_value += metric_value6
                if criterion_periodic:
                    metric_value7 = w7 * metric(predictions, predictions[:,model.nn_idx_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idx_2_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idy_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idy_2_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idz_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idz_2_full])
                    metric_value += metric_value7
            
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]

                running_loss1 += loss1.item() * X_batch.shape[0]
                running_metric1 += metric_value1 * X_batch.shape[0]

                running_loss2 += loss2.item() * X_batch.shape[0]
                running_metric2 += metric_value2 * X_batch.shape[0]

                running_loss3 += loss3.item() * X_batch.shape[0]
                running_metric3 += metric_value3 * X_batch.shape[0]

                running_loss4 += loss4.item() * X_batch.shape[0]
                running_metric4 += metric_value4 * X_batch.shape[0]
                
                running_loss5 += loss5.item() * X_batch.shape[0]
                running_metric5 += metric_value5 * X_batch.shape[0]

                running_loss6 += loss6.item() * X_batch.shape[0]
                running_metric6 += metric_value6 * X_batch.shape[0]

                running_loss7 += loss7.item() * X_batch.shape[0]
                running_metric7 += metric_value7 * X_batch.shape[0]

        val_losses += [running_loss / len(val_loader.dataset)]
        val_metrics += [running_metric / len(val_loader.dataset)]
        val_losses1 += [running_loss1 / len(val_loader.dataset)]
        val_metrics1 += [running_metric1 / len(val_loader.dataset)]
        val_losses2 += [running_loss2 / len(val_loader.dataset)]
        val_metrics2 += [running_metric2 / len(val_loader.dataset)]
        val_losses3 += [running_loss3 / len(val_loader.dataset)]
        val_metrics3 += [running_metric3 / len(val_loader.dataset)]
        val_losses4 += [running_loss4 / len(val_loader.dataset)]
        val_metrics4 += [running_metric4 / len(val_loader.dataset)]
        val_losses5 += [running_loss5 / len(val_loader.dataset)]
        val_metrics5 += [running_metric5 / len(val_loader.dataset)]
        val_losses6 += [running_loss6 / len(val_loader.dataset)]
        val_metrics6 += [running_metric6 / len(val_loader.dataset)]
        val_losses7 += [running_loss7 / len(val_loader.dataset)]
        val_metrics7 += [running_metric7 / len(val_loader.dataset)]
        # if verbose and i % 10 == 0:
        if run is not None:
            run.log(
                    # pbar.set_postfix(
                        {"val_loss": val_losses[-1], "val_MSE": val_metrics[-1],
                                     "val_loss1_init": val_losses1[-1], "val_MSE1_init": val_metrics1[-1],
                                      "val_loss2_sampling": val_losses2[-1], "val_MSE2_sampling": val_metrics2[-1],
                                      "val_loss3_pinn": val_losses3[-1], "val_MSE3_pinn": val_metrics3[-1],
                                      "val_loss4_n": val_losses4[-1], "val_MSE4_n": val_metrics4[-1],
                                      "val_loss5_e": val_losses5[-1], "val_MSE5_e": val_metrics5[-1],
                                      "val_loss6_pinn_model": val_losses6[-1], "val_MSE6_pinn_model": val_metrics6[-1],
                                      "val_loss7_boundary_periodic": val_losses7[-1], "val_MSE7_boundary_periodic": val_metrics7[-1],
                                     })#, step=epoch)

        if patience is not None:
            current_val_loss = val_losses[-1]
            if current_val_loss < best_val_loss - etol:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}: val_loss did not improve by more than {etol} for {patience} epochs.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    return train_metrics[-1], val_metrics[-1]



def UNet_train_and_validate(
    model,
    optimizer,
    criterion,
    metric,
    train_loader,
    val_loader,
    init_loader,
    num_epochs,
    vae=None,
    criterion_init_cond=False,
    criterion_ibound=False,
    criterion_pinn=False,
    criterion_Nconst=False,
    criterion_Econst=False,
    criterion_pinn_model=False,
    criterion_periodic=False,
    verbose=True,
    w1=1,
    w2=1,
    w3=1,
    w4=1,
    w5=1,
    w6=1,
    w7=1,
    device='cpu',
    run=None,
    patience=None,
    etol=1e-4,
    max_grad_norm=100.0
):
    """
    Train and validate neural network
      - model: neural network to train
      - optimizer: optimizer chained to a model
      - criterion: loss function class
      - metric: function to measure MSE taking neural networks predictions
                 and ground truth labels
      - train_loader: DataLoader with train set
      - val_loader: DataLoader with validation set
      - num_epochs: number of epochs to train
      - verbose: whether to plot metrics during training
    Returns:
      - train_mse: training MSE over the last epoch
      - val_mse: validation MSE after the last epoch
    """

    _t0_init, _psi0_init = next(iter(init_loader))
    init_loader = DataLoader(
        TensorDataset(_t0_init.to(device), _psi0_init.to(device)),
        batch_size=_t0_init.shape[0],
    )

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    train_losses1, val_losses1 = [], []
    train_metrics1, val_metrics1 = [], []
    train_losses2, val_losses2 = [], []
    train_metrics2, val_metrics2 = [], []
    train_losses3, val_losses3 = [], []
    train_metrics3, val_metrics3 = [], []
    train_losses4, val_losses4 = [], []
    train_metrics4, val_metrics4 = [], []
    train_losses5, val_losses5 = [], []
    train_metrics5, val_metrics5 = [], []
    train_losses6, val_losses6 = [], []
    train_metrics6, val_metrics6 = [], []
    train_losses7, val_losses7 = [], []
    train_metrics7, val_metrics7 = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        running_loss1, running_metric1 = 0, 0
        running_loss2, running_metric2 = 0, 0
        running_loss3, running_metric3 = 0, 0
        running_loss4, running_metric4 = 0, 0
        running_loss5, running_metric5 = 0, 0
        running_loss6, running_metric6 = 0, 0
        running_loss7, running_metric7 = 0, 0
        pbar = (
            tqdm(train_loader, desc=f"Training {epoch}/{num_epochs}")
            if verbose
            else train_loader
        )
        interm_losses = []
        interm_metrics = []

        pre_epoch_state = {k: v.clone() for k, v in model.state_dict().items()}
        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            def closure():
                t0, psi0 = next(iter(init_loader))
                X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                # Flatten y_batch to (B, 2*V) — UNet loaders return 5D targets
                # but the model always outputs flat. No-op for already-flat targets.
                y_batch_flat = y_batch.reshape(y_batch.shape[0], -1)
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    psi0 = psi0.reshape(psi0.shape[0], -1)
                    # print(model(t0).shape, psi0.shape)
                    loss1 =  w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-loss)])).requires_grad_(False)
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch_flat)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * loss1), torch.exp(-w2 * loss2)])).requires_grad_(False)
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch_flat))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch_flat))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch_flat))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = w7 * criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += loss7
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent explosion
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # If loss is NaN/Inf, zero out grads so LBFGS skips the update
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                return loss
            optimizer.step(closure)
            # optimizer.step()

            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                if verbose:
                    print(f"NaN detected in weights at epoch {epoch}, batch {i}. Rolling back to pre-epoch checkpoint.")
                model.load_state_dict(pre_epoch_state)
                break  # skip the rest of this epoch

            with torch.no_grad():
                t0, psi0 = next(iter(init_loader))
                X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                y_batch_flat = y_batch.reshape(y_batch.shape[0], -1)
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    psi0 = psi0.reshape(psi0.shape[0], -1)
                    # print(model(t0).shape, psi0.shape)
                    loss1 = w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-loss)])).requires_grad_(False)
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch_flat)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * loss1), torch.exp(-w2 * loss2)])).requires_grad_(False)
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch_flat))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch_flat))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch_flat))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = w7 * criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += w7 * criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += loss7
                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    psi0 = psi0.reshape(psi0.shape[0], -1)
                    metric_value1 = w1 * metric(model(t0), psi0)
                    metric_value = metric_value1
                    metric_value2, metric_value3, metric_value4, metric_value5, metric_value6, metric_value7 = metric_value1, metric_value1, metric_value1, metric_value1, metric_value1, metric_value1

                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-metric_value1)])).requires_grad_(False)
                if criterion_ibound:
                    metric_value2 = w2 * metric(predictions, y_batch_flat)
                    metric_value += metric_value2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * metric_value1), torch.exp(-w2 * metric_value2)])).requires_grad_(False)
                if criterion_pinn:
                    metric_value3 = w3 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch_flat))
                    metric_value += metric_value3
                if criterion_Nconst:
                    metric_value4 = w4 * metric(model.N(model(X_batch_aug)), model.N(y_batch_flat))
                    metric_value += metric_value4
                if criterion_Econst:
                    metric_value5 = w5 * metric(model.E(model(X_batch_aug)), model.E(y_batch_flat))
                    metric_value += metric_value5
                if criterion_pinn_model:
                    metric_value6 = w6 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    metric_value += metric_value6
                if criterion_periodic:
                    metric_value7 = w7 *metric(predictions, predictions[:,model.nn_idx_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idx_2_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idy_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idy_2_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idz_1_full])
                    metric_value7 += w7 *metric(predictions, predictions[:,model.nn_idz_2_full])
                    metric_value +=  metric_value7
            
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]
                interm_losses += [1. * running_loss / (i+1)]
                interm_metrics += [1. * running_metric / (i+1)]
                # print(interm_losses)
                running_loss1 += loss1.item() * X_batch.shape[0]
                running_metric1 += metric_value1 * X_batch.shape[0]

                running_loss2 += loss2.item() * X_batch.shape[0]
                running_metric2 += metric_value2 * X_batch.shape[0]

                running_loss3 += loss3.item() * X_batch.shape[0]
                running_metric3 += metric_value3 * X_batch.shape[0]

                running_loss4 += loss4.item() * X_batch.shape[0]
                running_metric4 += metric_value4 * X_batch.shape[0]
                
                running_loss5 += loss5.item() * X_batch.shape[0]
                running_metric5 += metric_value5 * X_batch.shape[0]

                running_loss6 += loss6.item() * X_batch.shape[0]
                running_metric6 += metric_value6 * X_batch.shape[0]

                running_loss7 += loss7.item() * X_batch.shape[0]
                running_metric7 += metric_value7 * X_batch.shape[0]


        train_losses += [running_loss / len(train_loader.dataset)]
        train_metrics += [running_metric / len(train_loader.dataset)]
        train_losses1 += [running_loss1 / len(train_loader.dataset)]
        train_metrics1 += [running_metric1 / len(train_loader.dataset)]
        train_losses2 += [running_loss2 / len(train_loader.dataset)]
        train_metrics2 += [running_metric2 / len(train_loader.dataset)]
        train_losses3 += [running_loss3 / len(train_loader.dataset)]
        train_metrics3 += [running_metric3 / len(train_loader.dataset)]
        train_losses4 += [running_loss4 / len(train_loader.dataset)]
        train_metrics4 += [running_metric4 / len(train_loader.dataset)]
        train_losses5 += [running_loss5 / len(train_loader.dataset)]
        train_metrics5 += [running_metric5 / len(train_loader.dataset)]
        train_losses6 += [running_loss6 / len(train_loader.dataset)]
        train_metrics6 += [running_metric6 / len(train_loader.dataset)]
        train_losses7 += [running_loss7 / len(train_loader.dataset)]
        train_metrics7 += [running_metric7 / len(train_loader.dataset)]
        if run is not None:
            run.log(
                    # pbar.set_postfix(
                        {"train_loss": train_losses[-1], "train_MSE": train_metrics[-1],
                                     "train_loss1_init": train_losses1[-1], "train_MSE1_init": train_metrics1[-1],
                                      "train_loss2_sampling": train_losses2[-1], "train_MSE2_sampling": train_metrics2[-1],
                                      "train_loss3_pinn": train_losses3[-1], "train_MSE3_pinn": train_metrics3[-1],
                                      "train_loss4_n": train_losses4[-1], "train_MSE4_n": train_metrics4[-1],
                                      "train_loss5_e": train_losses5[-1], "train_MSE5_e": train_metrics5[-1],
                                      "train_loss6_pinn_model": train_losses6[-1], "train_MSE6_pinn_model": train_metrics6[-1],
                                      "train_loss7_boundary_periodic": train_losses7[-1], "train_MSE7_boundary_periodic": train_metrics7[-1],
                                     })#, step=epoch)
        model.eval()
        running_loss, running_metric = 0, 0
        running_loss1, running_metric1 = 0, 0
        running_loss2, running_metric2 = 0, 0
        running_loss3, running_metric3 = 0, 0
        running_loss4, running_metric4 = 0, 0
        running_loss5, running_metric5 = 0, 0
        running_loss6, running_metric6 = 0, 0
        running_loss7, running_metric7 = 0, 0
        pbar = (
            tqdm(val_loader, desc=f"Validating {epoch}/{num_epochs}")
            if verbose
            else val_loader
        )

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                t0, psi0 = next(iter(init_loader))
                X_batch_aug = X_batch
                predictions = model.forward(X_batch_aug)
                y_batch_flat = y_batch.reshape(y_batch.shape[0], -1)

                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    psi0 = psi0.reshape(psi0.shape[0], -1)
                    loss1 =  w1 * criterion(model(t0), psi0)
                    loss2, loss3, loss4, loss5, loss6, loss7 = loss1, loss1, loss1, loss1, loss1, loss1
                    loss = loss1
                    #normalize by t=0
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-w1*loss1)]))
                if criterion_ibound:
                    loss2 = w2 * criterion(predictions, y_batch_flat)
                    loss += loss2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1*loss1), torch.exp(-w2 * loss2)]))
                if criterion_pinn:
                    loss3 = w3 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch_flat))
                    loss += loss3
                if criterion_Nconst:
                    loss4 = w4 * criterion(model.N(model(X_batch_aug)), model.N(y_batch_flat))
                    loss += loss4
                if criterion_Econst:
                    loss5 = w5 * criterion(model.E(model(X_batch_aug)), model.E(y_batch_flat))
                    loss += loss5
                if criterion_pinn_model:
                    loss6 = w6 * criterion(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    loss += loss6
                if criterion_periodic:
                    loss7 = criterion(predictions, predictions[:,model.nn_idx_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idx_2_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idy_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idy_2_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idz_1_full])
                    loss7 += criterion(predictions, predictions[:,model.nn_idz_2_full])
                    loss += w7 * loss7

                if criterion_init_cond:
                    t0, psi0 = next(iter(init_loader))
                    psi0 = psi0.reshape(psi0.shape[0], -1)
                    metric_value1 = w1 * metric(model(t0), psi0)
                    metric_value = metric_value1
                    metric_value2, metric_value3, metric_value4, metric_value5, metric_value6, metric_value7 = metric_value1, metric_value1, metric_value1, metric_value1, metric_value1, metric_value1
                    # w2 = torch.min(torch.FloatTensor([1, torch.exp(-metric_value1)])).requires_grad_(False)
                if criterion_ibound:
                    metric_value2 = w2 * metric(predictions, y_batch_flat)
                    metric_value += metric_value2
                    # w3 = torch.min(torch.FloatTensor([torch.exp(-w1 * metric_value1), torch.exp(-w2 * metric_value2)])).requires_grad_(False)
                if criterion_pinn:
                    metric_value3 = w3 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(y_batch_flat))
                    metric_value += metric_value3
                if criterion_Nconst:
                    metric_value4 = w4 * metric(model.N(model(X_batch_aug)), model.N(y_batch_flat))
                    metric_value += metric_value4
                if criterion_Econst:
                    metric_value5 = w5 * metric(model.E(model(X_batch_aug)), model.E(y_batch_flat))
                    metric_value += metric_value5
                if criterion_pinn_model:
                    metric_value6 = w6 * metric(model.dpsi_dt(X_batch_aug), model.dpsi_dt_fn(model.dpsi_dt(X_batch_aug)))
                    metric_value += metric_value6
                if criterion_periodic:
                    metric_value7 = w7 * metric(predictions, predictions[:,model.nn_idx_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idx_2_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idy_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idy_2_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idz_1_full])
                    metric_value7 += w7 * metric(predictions, predictions[:,model.nn_idz_2_full])
                    metric_value += metric_value7
            
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]

                running_loss1 += loss1.item() * X_batch.shape[0]
                running_metric1 += metric_value1 * X_batch.shape[0]

                running_loss2 += loss2.item() * X_batch.shape[0]
                running_metric2 += metric_value2 * X_batch.shape[0]

                running_loss3 += loss3.item() * X_batch.shape[0]
                running_metric3 += metric_value3 * X_batch.shape[0]

                running_loss4 += loss4.item() * X_batch.shape[0]
                running_metric4 += metric_value4 * X_batch.shape[0]
                
                running_loss5 += loss5.item() * X_batch.shape[0]
                running_metric5 += metric_value5 * X_batch.shape[0]

                running_loss6 += loss6.item() * X_batch.shape[0]
                running_metric6 += metric_value6 * X_batch.shape[0]

                running_loss7 += loss7.item() * X_batch.shape[0]
                running_metric7 += metric_value7 * X_batch.shape[0]

        val_losses += [running_loss / len(val_loader.dataset)]
        val_metrics += [running_metric / len(val_loader.dataset)]
        val_losses1 += [running_loss1 / len(val_loader.dataset)]
        val_metrics1 += [running_metric1 / len(val_loader.dataset)]
        val_losses2 += [running_loss2 / len(val_loader.dataset)]
        val_metrics2 += [running_metric2 / len(val_loader.dataset)]
        val_losses3 += [running_loss3 / len(val_loader.dataset)]
        val_metrics3 += [running_metric3 / len(val_loader.dataset)]
        val_losses4 += [running_loss4 / len(val_loader.dataset)]
        val_metrics4 += [running_metric4 / len(val_loader.dataset)]
        val_losses5 += [running_loss5 / len(val_loader.dataset)]
        val_metrics5 += [running_metric5 / len(val_loader.dataset)]
        val_losses6 += [running_loss6 / len(val_loader.dataset)]
        val_metrics6 += [running_metric6 / len(val_loader.dataset)]
        val_losses7 += [running_loss7 / len(val_loader.dataset)]
        val_metrics7 += [running_metric7 / len(val_loader.dataset)]
        # if verbose and i % 10 == 0:
        if run is not None:
            run.log(
                    # pbar.set_postfix(
                        {"val_loss": val_losses[-1], "val_MSE": val_metrics[-1],
                                     "val_loss1_init": val_losses1[-1], "val_MSE1_init": val_metrics1[-1],
                                      "val_loss2_sampling": val_losses2[-1], "val_MSE2_sampling": val_metrics2[-1],
                                      "val_loss3_pinn": val_losses3[-1], "val_MSE3_pinn": val_metrics3[-1],
                                      "val_loss4_n": val_losses4[-1], "val_MSE4_n": val_metrics4[-1],
                                      "val_loss5_e": val_losses5[-1], "val_MSE5_e": val_metrics5[-1],
                                      "val_loss6_pinn_model": val_losses6[-1], "val_MSE6_pinn_model": val_metrics6[-1],
                                      "val_loss7_boundary_periodic": val_losses7[-1], "val_MSE7_boundary_periodic": val_metrics7[-1],
                                     })#, step=epoch)

        if patience is not None:
            current_val_loss = val_losses[-1]
            if current_val_loss < best_val_loss - etol:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}: val_loss did not improve by more than {etol} for {patience} epochs.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    return train_metrics[-1], val_metrics[-1]


