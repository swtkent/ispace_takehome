import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import lunar_transfer_utils as ltu

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
    """

    def objective(dv):
        # objective function to put into optimizing function to minimize DV
        if isinstance(dv, np.ndarray) and dv.shape==(1,):
            dv = dv[0]
        sep = separation_after_one_revolution(dv, a_initial, e_initial,
                                                M_start, mu)
        return abs(sep - target_separation)

    result = minimize(objective, x0=init_guess)#, bounds=[(-bound, bound)])
    optimal_dv = result.x[0]
    min_val_found = result.fun

    ta = ltu.true_anomaly_from_mean_anomaly(M_start, e_initial)
    r_deploy = ltu.radius_at_true_anomaly(a_initial, e_initial, ta)
    init_v = ltu.velocity_at_radius(a_initial, r_deploy, mu)
    separation = separation_after_one_revolution(optimal_dv, a_initial, e_initial,
                                                M_start, mu)
    # to help with debugging
    print(f"{M_start=:.3f} rads\t{optimal_dv=:.4f} m/s\t{separation=:.3f} m")

    return optimal_dv


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

    # Optimized Delta-V's at peri- and apo- apses
    dv_perilune = optimize_dv(a, e, 0, ltu.MU_MOON, target_separation, init_guess,
                              bound)  # Velocity at periapsis
    dv_apolune = optimize_dv(a, e, np.pi, ltu.MU_MOON, target_separation,
                             init_guess, bound)  # Velocity at apoapsis
    print(f"Minimum Delta-V at Perilune: {dv_perilune:.3f} m/s")
    print(f"Minimum Delta-V at Apolune: {dv_apolune:.3f} m/s")


if __name__ == "__main__":
    main()
