import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent / "DGPE"))
from GPElib.dynamics_generator import DynamicsGenerator


def generate_single_simulation(
    seed: int,
    N_wells: tuple = (10, 10, 10),
    time: float = 51.0,
    step: float = 0.05,
    beta: float = 10.0,
    energy_per_site: float = 1.0,
    save_path: Path = None,
    **dgpe_kwargs
):
    """
    Generate a single GPE simulation with random initial conditions.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    N_wells : tuple
        Grid dimensions (Nx, Ny, Nz)
    time : float
        Total simulation time
    step : float
        Time step for integration
    beta : float
        Nonlinearity parameter
    energy_per_site : float
        Initial energy per lattice site
    save_path : Path
        Path to save the simulation results
    **dgpe_kwargs : dict
        Additional keyword arguments for DynamicsGenerator

    Returns
    -------
    dict
        Dictionary containing simulation results and metadata
    """

    dgpe = DynamicsGenerator(
        N_part_per_well=1.0,
        N_wells=N_wells,
        dimensionality=3,
        anisotropy=1.0,
        threshold_XY_to_polar=0.25,
        beta=beta,
        local_disorder_amplitude=0.00,
        FloatPrecision=np.float64,
        integration_method='RK45',
        rtol=1e-8,
        atol=1e-8,
        smooth_quench=True,
        reset_steps_duration=5,
        calculation_type='lyap_save_all',
        integrator='scipy',
        time=time,
        step=step,
        gamma=1.0,
        traj_seed=seed,
        disorder_seed=53,
        **dgpe_kwargs
    )

    dgpe.generate_init(
        traj_seed=seed,
        energy_per_site=energy_per_site,
        kind='random_population_and_phase'
    )

    dgpe.set_init_XY(dgpe.X[:, :, :, 0], dgpe.Y[:, :, :, 0])
    dgpe.icurr = 0
    dgpe.inext = 1

    dgpe.run_dynamics(no_pert=False)

    results = {
        'X': dgpe.X,  # Real part of wavefunction
        'Y': dgpe.Y,  # Imaginary part of wavefunction
        'energy': dgpe.energy,
        'participation_rate': dgpe.participation_rate,
        'number_of_particles': dgpe.number_of_particles,
        'metadata': {
            'seed': seed,
            'N_wells': N_wells,
            'time': time,
            'step': step,
            'beta': beta,
            'energy_per_site': energy_per_site,
            'timestamp': datetime.now().isoformat()
        }
    }

    # Save if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **results)

    return results


def generate_gpe_dataset(
    n_samples: int = 1000,
    output_dir: str = "datasets/gpe_simulations",
    N_wells: tuple = (10, 10, 10),
    time: float = 51.0,
    step: float = 0.05,
    beta: float = 10.0,
    energy_per_site: float = 1.0,
    start_seed: int = 0,
    parallel: bool = False,
    n_jobs: int = -1,
    **dgpe_kwargs
):
    """
    Generate a dataset of GPE simulations with random initial conditions.

    Parameters
    ----------
    n_samples : int
        Number of simulations to generate
    output_dir : str
        Directory to save the dataset
    N_wells : tuple
        Grid dimensions (Nx, Ny, Nz)
    time : float
        Total simulation time
    step : float
        Time step for integration
    beta : float
        Nonlinearity parameter
    energy_per_site : float
        Initial energy per lattice site
    start_seed : int
        Starting seed for random number generation
    parallel : bool
        Whether to use parallel processing
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    **dgpe_kwargs : dict
        Additional keyword arguments for DynamicsGenerator
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        'n_samples': n_samples,
        'N_wells': N_wells,
        'time': time,
        'step': step,
        'beta': beta,
        'energy_per_site': energy_per_site,
        'start_seed': start_seed,
        'created_at': datetime.now().isoformat()
    }
    np.savez(output_path / 'dataset_metadata.npz', **metadata)

    print(f"Generating {n_samples} GPE simulations...")
    print(f"Grid size: {N_wells}")
    print(f"Output directory: {output_path}")

    if parallel:
        from joblib import Parallel, delayed

        def generate_and_save(i):
            seed = start_seed + i
            save_path = output_path / f"sim_{seed:06d}.npz"
            if not save_path.exists():
                generate_single_simulation(
                    seed=seed,
                    N_wells=N_wells,
                    time=time,
                    step=step,
                    beta=beta,
                    energy_per_site=energy_per_site,
                    save_path=save_path,
                    **dgpe_kwargs
                )
            return seed

        Parallel(n_jobs=n_jobs)(
            delayed(generate_and_save)(i)
            for i in tqdm(range(n_samples), desc="Generating simulations")
        )
    else:
        for i in tqdm(range(n_samples), desc="Generating simulations"):
            seed = start_seed + i
            save_path = output_path / f"sim_{seed:06d}.npz"

            if save_path.exists():
                print(f"Skipping seed {seed} (already exists)")
                continue

            generate_single_simulation(
                seed=seed,
                N_wells=N_wells,
                time=time,
                step=step,
                beta=beta,
                energy_per_site=energy_per_site,
                save_path=save_path,
                **dgpe_kwargs
            )

    print(f"\nDataset generation complete!")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPE simulation dataset"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of simulations to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/gpe_simulations",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="Grid dimensions (Nx Ny Nz)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=51.0,
        help="Simulation time"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Time step"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Nonlinearity parameter"
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=1.0,
        help="Initial energy per site"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting random seed"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs"
    )

    args = parser.parse_args()

    generate_gpe_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        N_wells=tuple(args.grid_size),
        time=args.time,
        step=args.step,
        beta=args.beta,
        energy_per_site=args.energy,
        start_seed=args.start_seed,
        parallel=args.parallel,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()
