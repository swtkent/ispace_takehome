import numpy as np
from scipy.optimize import minimize

# Assumed Constants
MU_MOON = 4.9048695e12  # Gravitational parameter of Moon (m^3/s^2)
MOON_RADIUS = 1737.4e3  # Radius of Moon (m)

# Initial orbit parameters
PERILUNE_ALT = 100e3  # m
APOLUNE_ALT = 10000e3  # m
TARGET_SEPARATION = 10e3 # m


def orbital_period(a: float, mu: float) -> float:
    """
    Compute the orbital period around an object with SGP value of mu and a semi-major axis of a

    Parameters
    a: Semi-major axis of the initial orbit (m)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    period: Time to complete one orbit revolution (s)
    """
    
    period = 2 * np.pi * np.sqrt(a**3 / mu) # s
    return period


def velocity_at_radius(r: float, a: float, mu: float) -> float:    
    """
    Compute orbital velocity at a distance r from the center for an orbit with semi-major axis a.

    Parameters
    r: Radius of satellite (m)
    a: Semi-major axis of the initial orbit (m)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    velocity: Velocity at given radius for orbit with semi-major axis a and SGP mu (m/s)
    """

    velocity = np.sqrt(mu * (2 / r - 1 / a))
    return velocity


def sma_at_rv(r: float, v: float, mu: float) -> float:
    """
    Copmute the semi-major axis given distance r and velocity v aroundd central body with SGP mu
    Reworked equation for solving velocity with same parameters

    Parameters
    r: Radius of satellite (m)
    v: Velocity of satellite (m/s)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    a: Semi-major axis of the orbit (m)
    """
    a = 1 / (2 / r - v**2 / mu)
    return a


def separation_distance(a_initial:float, a_new: float, r_deploy: float) -> float:
    """
    Compute the position difference after one revolution

    Parameters
    a_initial: initial semi-major axis before separation (m)
    a_new: new semi-major axis after separation (m)
    r_deploy: the radius at which deployed from

    Returns
    separation: separation distance
    """
    # Compute position difference after one revolution
    theta_initial = 2 * np.pi
    theta_new = theta_initial * (np.sqrt(a_initial**3 / a_new**3))
    delta_theta = np.abs(theta_initial - theta_new)
    separation = np.sqrt((r_deploy * np.sin(delta_theta))**2 + (r_deploy * (1 - np.cos(delta_theta)))**2)
    return separation


def separation_after_one_revolution(dv: float, r: float, a_initial: float, mu: float) -> float:
    """
    Compute the separation distance after one orbital revolution

    Parameters
    dv: Delta-V for the separation (m/s)
    r: radius at which separated (m)
    v: velocity at radius at which separated (m/s)
    a_inital: Semi-major axis of original orbit (m)
    mu: Standard graviational parameter (m^3/s^2)

    Returns
    separation: Separation distance after one orbit (m)
    """

    # Calculate original orbital period
    T = orbital_period(a_initial, mu)

    # Calculate the velocity at given position r
    v = velocity_at_radius(r, a_initial, mu)

    # Deployed satellite velocity
    v_deployed = v + dv

    # New semi-major axis and orbital period
    a_new = sma_at_rv(r, v_deployed, mu)
    T_new = orbital_period(a_new, mu)

    # Relative separation after one period
    separation = separation_distance(a_initial, a_new, r)
    return separation


def optimize_dv(a_initial: float, r_deploy: float, target_separation: float, mu: float) -> float:
    """
    Compute the minimum Delta-V to achieve the target separation after one revolution.

    Parameters
    a_initial: Semi-major axis of the initial orbit (m)
    r_deploy: Deployment radius (m)
    target_separation: Target separation distance (m)
    mu: Standard Gravitational Parameter (m^3/s^2)

    Returns
    optimal_dv: Optimal delta-V (m/s)
    """

    def objective(dv):
        # objective function to put into optimizing function to minimize delta v
        return abs(separation_after_one_revolution(dv, a_initial, r_deploy, mu) - target_separation)

    result = minimize(objective, x0=40.0, bounds=[(0, 1e3)])
    optimal_dv = result.x[0]
    return optimal_dv


def main():

    # Initial orbit parameters
    a = (MOON_RADIUS + PERILUNE_ALT + MOON_RADIUS + APOLUNE_ALT) / 2  # Semi-major axis
    r_p = MOON_RADIUS + PERILUNE_ALT  # Periapsis radius
    r_a = MOON_RADIUS + APOLUNE_ALT  # Apoapsis radius

    dv_perilune = optimize_dv(a, r_p, TARGET_SEPARATION, MU_MOON)  # Velocity at periapsis
    dv_apolune = optimize_dv(a, r_a, TARGET_SEPARATION, MU_MOON)  # Velocity at apoapsis


    print(f"Minimum Delta-V at Perilune: {dv_perilune:.3f} m/s")
    print(f"Minimum Delta-V at Apolune: {dv_apolune:.3f} m/s")


if __name__=="__main__":
    main()