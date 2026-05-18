import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json


class GPEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "initial_conditions",
        time_steps: Optional[List[int]] = None,
        transform: Optional[callable] = None,
        normalize: bool = True,
        cache_in_memory: bool = False
    ):
        """
        Initialize GPEDataset.

        Parameters
        ----------
        data_dir : str
            Directory containing the simulation .npz files
        mode : str
            What data to return:
            - 'initial_conditions': Return only initial conditions (for VAE)
            - 'trajectories': Return full trajectories (for sequence models)
            - 'snapshots': Return individual time snapshots
        time_steps : List[int], optional
            Specific time steps to include (None = all)
        transform : callable, optional
            Transform to apply to the data
        normalize : bool
            Whether to normalize the data
        cache_in_memory : bool
            Whether to cache all data in memory (faster but uses more RAM)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.time_steps = time_steps
        self.transform = transform
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        self.file_paths = sorted(self.data_dir.glob("sim_*.npz"))

        if len(self.file_paths) == 0:
            raise ValueError(f"No simulation files found in {data_dir}")

        print(f"Found {len(self.file_paths)} simulation files")

        metadata_path = self.data_dir / "dataset_metadata.npz"
        if metadata_path.exists():
            self.metadata = dict(np.load(metadata_path, allow_pickle=True))
            print(f"Dataset metadata loaded")
        else:
            self.metadata = {}
            print("Warning: No dataset metadata found")

        if self.normalize:
            self._compute_normalization_stats()

        self.cache = {}
        if self.cache_in_memory:
            print("Caching data in memory...")
            for idx in range(len(self.file_paths)):
                self.cache[idx] = self._load_simulation(idx)
            print("Data cached successfully")

    def _compute_normalization_stats(self, sample_size: int = 100):
        print("Computing normalization statistics...")

        sample_indices = np.random.choice(
            len(self.file_paths),
            size=min(sample_size, len(self.file_paths)),
            replace=False
        )

        all_data = []
        for idx in sample_indices:
            data = np.load(self.file_paths[idx])
            X = data['X'][:, :, :, 0]  # Initial condition
            Y = data['Y'][:, :, :, 0]
            psi = np.stack([X, Y], axis=0)  # Shape: (2, Nx, Ny, Nz)
            all_data.append(psi)

        all_data = np.array(all_data)  # Shape: (N, 2, Nx, Ny, Nz)

        # Compute statistics per channel (average over batch and spatial dimensions)
        # Result shape after mean: (1, 2, 1, 1, 1) -> reshape to (2, 1, 1, 1)
        self.mean = all_data.mean(axis=(0, 2, 3, 4), keepdims=True).squeeze(0)  # Shape: (2, 1, 1, 1)
        self.std = all_data.std(axis=(0, 2, 3, 4), keepdims=True).squeeze(0) + 1e-8  # Shape: (2, 1, 1, 1)

        print(f"Normalization stats computed: mean shape {self.mean.shape}, std shape {self.std.shape}")

    def _load_simulation(self, idx: int) -> Dict:
        data = dict(np.load(self.file_paths[idx], allow_pickle=True))
        return data

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            # mean and std have shape (2, 1, 1, 1), x has shape (2, Nx, Ny, Nz) or (2, Nx, Ny, Nz, T)
            return (x - self.mean) / self.std
        return x

    def __len__(self) -> int:
        if self.mode == 'snapshots' and self.time_steps is not None:
            return len(self.file_paths) * len(self.time_steps)
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single sample from the dataset.

        Returns
        -------
        data : torch.Tensor
            The simulation data
        metadata : dict
            Metadata for this sample
        """
        if self.cache_in_memory and idx in self.cache:
            sim_data = self.cache[idx]
        else:
            sim_data = self._load_simulation(idx)

        X = sim_data['X']
        Y = sim_data['Y']

        if self.mode == 'initial_conditions':
            # Return only initial conditions (t=0)
            # Shape: (2, Nx, Ny, Nz)
            psi = np.stack([X[:, :, :, 0], Y[:, :, :, 0]], axis=0)
            psi = self._normalize(psi)
            psi_tensor = torch.from_numpy(psi).float()

            metadata = {
                'idx': idx,
                'seed': sim_data['metadata'].item()['seed'] if 'metadata' in sim_data else idx
            }

        elif self.mode == 'trajectories':
            # Return full trajectory
            # Shape: (2, Nx, Ny, Nz, T)
            if self.time_steps is not None:
                X = X[:, :, :, self.time_steps]
                Y = Y[:, :, :, self.time_steps]

            psi = np.stack([X, Y], axis=0)
            psi = self._normalize(psi)
            psi_tensor = torch.from_numpy(psi).float()

            metadata = {
                'idx': idx,
                'seed': sim_data['metadata'].item()['seed'] if 'metadata' in sim_data else idx,
            }
            if 'energy' in sim_data:
                metadata['energy'] = torch.from_numpy(sim_data['energy']).float()

        elif self.mode == 'snapshots':
            # Return a single time snapshot
            if self.time_steps is None:
                # Random time step
                t_idx = np.random.randint(0, X.shape[-1])
            else:
                file_idx = idx // len(self.time_steps)
                t_idx = self.time_steps[idx % len(self.time_steps)]
                if not self.cache_in_memory:
                    sim_data = self._load_simulation(file_idx)
                    X = sim_data['X']
                    Y = sim_data['Y']

            psi = np.stack([X[:, :, :, t_idx], Y[:, :, :, t_idx]], axis=0)
            psi = self._normalize(psi)
            psi_tensor = torch.from_numpy(psi).float()

            metadata = {
                'idx': idx,
                'time_idx': t_idx,
                'seed': sim_data['metadata'].item()['seed'] if 'metadata' in sim_data else idx
            }

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.transform is not None:
            psi_tensor = self.transform(psi_tensor)

        return psi_tensor, metadata


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    mode: str = "initial_conditions",
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Directory containing the dataset
    batch_size : int
        Batch size for DataLoader
    train_split : float
        Fraction of data for training
    val_split : float
        Fraction of data for validation
    mode : str
        Dataset mode
    num_workers : int
        Number of worker processes for data loading
    **dataset_kwargs
        Additional arguments for GPEDataset

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    """
    full_dataset = GPEDataset(data_dir=data_dir, mode=mode, **dataset_kwargs)

    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset split: {n_train} train, {n_val} val, {n_test} test")

    return train_loader, val_loader, test_loader
