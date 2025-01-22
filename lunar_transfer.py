import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import lunar_transfer_utils as ltu
from numpy.typing import NDArray
from typing import Tuple

MU_EARTH = 3.986e14  # (m^3/s^2)


def separation_after_one_revolution(dv: float, a_initial: float,
                                    e_initial: float, M_start: float,
                                    mu: float) -> float:
    """
    Compute the separation distance after one orbital revolution

    Parameters
    dv: Delta-V for the separation (m/s)
    a_inital: Semi-major axis of original orbit (m)
    e: Eccentricity (unitless)
    M_start: Mean anomaly at separation time (rads)
    mu: Standard graviational parameter (m^3/s^2)

    Returns
    separation: Separation distance after one orbit (m)
    """

    # Calculate original orbital period
    T = ltu.period_from_sma(a_initial, mu)

    # Relative separation after one period
    separation = ltu.separation_after_time(dv, a_initial, e_initial, M_start,
                                       T, mu)
    return separation


def optimize_dv(a_initial: float, e_initial: float, M_start: float, mu: float,
                target_separation: float, init_guess: float,
                bound: float) -> float:
    """
    Compute the minimum Delta-V to achieve the target separation after
    one revolution.

    Parameters
    a_initial: Semi-major axis of the initial orbit (m)
    e_initial: Eccentricity of initial orbit (unitless)
    M: Mean anomaly at separation time (rads)
    mu: Standard Gravitational Parameter (m^3/s^2)
    target_separation: Target separation distance (m)
    init_guess: Initial value to assume for delta v
    bound: upper value for positive/negative bounding of optimizer

    Returns
    optimal_dv: Optimal Delta-V (m/s)
    separation_val: Separation from optimal DV (m)
    """

    def objective(dv):
        # objective function to put into optimizing function to minimize DV
        if isinstance(dv, np.ndarray) and dv.shape==(1,):
            dv = dv[0]
        sep = separation_after_one_revolution(dv, a_initial, e_initial,
                                                M_start, mu)
        return abs(sep - target_separation)

    result = minimize(objective, x0=init_guess, bounds=[(-bound, bound)])

    if not result.success:
        return np.nan, np.nan
    
    optimal_dv = result.x[0]
    min_val_found = result.fun
    separation_val = min_val_found + target_separation
    return optimal_dv, separation_val


def find_successful_separation_dvs(a: float, e: float, M_range: NDArray[np.float64], target_separation: float, init_guess: float, bound: float) -> NDArray[np.float64]:
    """
    Compute the minimum Delta-V to achieve the target separation after
    one revolution for a series of mean anomalies.

    Parameters
    a: Semi-major axis of the initial orbit (m)
    e: Eccentricity of initial orbit (unitless)
    M_range: Range of mean anomalies (rads)
    target_separation: Target separation distance (m)
    init_guess: Initial value to assume for delta v
    bound: upper value for positive/negative bounding of optimizer

    Returns
    dvs: Optimal Delta-V (m/s)
    """
    dvs = np.empty_like(M_range)
    true_seps = np.empty_like(M_range)
    for i, M in enumerate(M_range):
        dvs[i], true_seps[i] = optimize_dv(a, e, M, ltu.MU_MOON, target_separation, init_guess, bound)
    target_errs = abs(true_seps - target_separation)
    return dvs, target_errs


def plot_optimal_dvs(M_range: NDArray[np.float64], dvs: NDArray[np.float64]) -> None:
    """
    Plot the optimal DV's found for each mean anomaly tested and return the smallest DV found and where

    Parameters
    M_range: Mean anomaly range (rads)
    dvs: Optimal Delta-V's (m/s)

    Returns
    None
    """
    # Find the index of the minimum absolute value
    min_index = np.nanargmin(np.abs(dvs))
    best_dv = dvs[min_index]
    best_M = M_range[min_index]

    plt.figure()
    plt.plot(M_range, dvs, color='blue', marker='.', markersize=4)
    plt.xlabel('Mean Anomaly (rads)')
    plt.ylabel('Delta-V (m/s)')
    plt.title('Optimal Delta-V for Targeted Separation')
    plt.scatter(best_M, best_dv, label=f'Minimal DV={best_dv:.4f} m/s at M={best_M:.4f}', marker='*', color='red', s=50)
    plt.grid()
    plt.legend()


def main():
    # Given initial orbit parameters
    perilune_alt = 100e3  # m
    apolune_alt = (10e3)*1e3  # m
    target_separation = 10e3  # m

    # values for guessing Delta V
    init_guess = 0  # m/s
    bound = 50  # m/s

    # Calculated initial orbit parameters
    a, e = ltu.orbit_shape_from_altitudes(perilune_alt, apolune_alt, ltu.MOON_RADIUS)

    # Optimized Delta-V's across orbit, that don't fail
    M_range = np.linspace(0, 2*np.pi, 10000)
    dvs, separation_errs = find_successful_separation_dvs(a, e, M_range, target_separation, init_guess, bound)
    plot_optimal_dvs(M_range, dvs)
    plt.show()


if __name__ == "__main__":
    main()
