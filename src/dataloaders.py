import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


def _to_float_tensor(array_like):
    return torch.as_tensor(array_like, dtype=torch.float32)

def generate_datasets(X, y, X_train, y_train, X_test, y_test, X_val, y_val, batch_size=64, istride=1):
    train_dataset = TensorDataset(torch.FloatTensor(X_train[::istride]), torch.FloatTensor(y_train[::istride]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
    init_dataset = TensorDataset(torch.FloatTensor(X[0,:]).reshape(1,-1), torch.FloatTensor(y[0,:]).reshape(1,-1))
    init_loader = torch.utils.data.DataLoader(init_dataset, batch_size)
    return train_loader, test_loader, val_loader, init_loader

def UNet_generate_datasets(X, y, X_train, y_train, X_test, y_test, X_val, y_val, batch_size=64, istride=1, step=0.1):
    def _t(a):
        return torch.as_tensor(a, dtype=torch.float32)

    y_train_t = _t(y_train); y_test_t = _t(y_test); y_val_t = _t(y_val); y_t = _t(y)
    spatial = y_train_t.shape[2:]
    
    dt_map_train = _t(X_train[istride::istride] - X_train[:-istride:istride]).view(-1, 1, 1, 1, 1).expand(-1, 1, *spatial)
    train_dataset = TensorDataset(torch.cat([y_train_t[:-istride:istride], dt_map_train], dim=1), y_train_t[istride::istride])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)

    dt_map_test = _t(X_test[istride::istride] - X_test[:-istride:istride]).view(-1, 1, 1, 1, 1).expand(-1, 1, *spatial)
    test_dataset = TensorDataset(torch.cat([y_test_t[:-istride:istride], dt_map_test], dim=1), y_test_t[istride::istride])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    dt_map_val = _t(X_val[istride::istride] - X_val[:-istride:istride]).view(-1, 1, 1, 1, 1).expand(-1, 1, *spatial)
    val_dataset = TensorDataset(torch.cat([y_val_t[:-istride:istride], dt_map_val], dim=1), y_val_t[istride::istride])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)

    dt_map_init = _t(X_val[0] - X_val[0]).view(-1, 1, 1, 1, 1).expand(-1, 1, *spatial)
    init_dataset = TensorDataset(torch.cat([y_t[0].unsqueeze(0), dt_map_init], dim=1), y_t[0].unsqueeze(0))
    init_loader = torch.utils.data.DataLoader(init_dataset, batch_size)

    return train_loader, test_loader, val_loader, init_loader


def UNet_generate_datasets_multi(
    X_list, y_list, train_idx, val_idx, test_idx,
    batch_size=64, istride=1,
):
    """
    Multi-IC version of UNet_generate_datasets.

    Args:
        X_list    : list of (T,) float arrays — time grids, one per IC
        y_list    : list of (T, 2, Nx, Ny, Nz) float arrays — states, one per IC
        train_idx, val_idx, test_idx : index lists into X_list / y_list
    """
    def _t(a):
        return torch.as_tensor(a, dtype=torch.float32)

    def _make_pairs(X_arr, y_arr):
        y_t = _t(y_arr)
        spatial = y_t.shape[2:]
        dt_map = _t(X_arr[istride::istride] - X_arr[:-istride:istride]) \
                     .view(-1, 1, 1, 1, 1).expand(-1, 1, *spatial).contiguous()
        return torch.cat([y_t[:-istride:istride], dt_map], dim=1), y_t[istride::istride]

    def _concat(indices):
        pairs = [_make_pairs(X_list[i], y_list[i]) for i in indices]
        return (torch.cat([p[0] for p in pairs]),
                torch.cat([p[1] for p in pairs]))

    X_tr, y_tr = _concat(train_idx)
    X_va, y_va = _concat(val_idx)
    X_te, y_te = _concat(test_idx)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size)

    # init_loader: t=0 state of every training IC (dt=0)
    spatial = _t(y_list[train_idx[0]]).shape[2:]
    init_X = torch.cat([
        torch.cat([_t(y_list[i][0]).unsqueeze(0),
                   torch.zeros(1, 1, *spatial, dtype=torch.float32)], dim=1)
        for i in train_idx
    ])
    init_y = torch.cat([_t(y_list[i][0]).unsqueeze(0) for i in train_idx])
    init_loader = DataLoader(TensorDataset(init_X, init_y), batch_size)

    return train_loader, test_loader, val_loader, init_loader
