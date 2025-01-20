import numpy as np
from numpy.typing import NDArray

# Assumed Constants
MU_MOON = 4.9048695e12  # Gravitational parameter of Moon (m^3/s^2)
MOON_RADIUS = 1737.4e3  # Radius of Moon (m)


def orbit_shape_from_altitudes(h_p, h_a, radius):
    """
    Compute the semi-major axis and eccentricity given peri- and apo- apsis
    altitudes and celestial body's radius

    Parameters
    h_p: Periapsis altitude (m)
    h_a: Apoapsis altitude (m)
    radius: Celestial body radius (m)

    Returns
    a: Semi-major axis (m)
    e: Eccentricity (unitless)
    """
    r_p = radius + h_p  # Periapsis radius
    r_a = radius + h_a  # Apoapsis radius
    a = (r_p + r_a) / 2  # Semi-major axis
    e = 1 - r_p / a  # Eccentricity
    return a, e


def mean_motion(a: float, mu: float) -> float:
    """
    Compute the mean motion given semi-major axis and standard
    gravitational parameter

    Parameters
    a: Semi-major axis of the initial orbit (m)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    n: Mean motion of orbit (rad/s)
    """
    n = np.sqrt(mu / a**3)
    return n


def period_from_mm(n: float) -> float:
    """
    Compute period from mean motion

    Parameters
    n: Mean motion of orbit (rad/s)

    Returns
    period: Time to complete one orbit revolution (s)
    """
    period = 2 * np.pi / n
    return period


def period_from_sma(a: float, mu: float) -> float:
    """
    Compute the orbital period around an object with SGP value of mu and
    semi-major axis of a

    Parameters
    a: Semi-major axis of the initial orbit (m)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    period: Time to complete one orbit revolution (s)
    """
    n = mean_motion(a, mu)
    period = period_from_mm(n)
    return period


def velocity_at_radius(r: float, a: float, mu: float) -> float:
    """
    Compute orbital velocity at a distance r from the center for an orbit with
    semi-major axis a

    Parameters
    r: Radius of satellite (m)
    a: Semi-major axis of the initial orbit (m)
    mu: Standard gravitational parameter (SGP) (m^3/s^2)

    Returns
    velocity: Velocity at given radius for orbit with semi-major axis a and
        SGP mu (m/s)
    """
    specific_energy = -mu / (2 * a)
    velocity = np.sqrt(2 * (specific_energy + mu / r))
    return velocity


def eccentric_anomaly_from_mean_anomaly(M, e, tol=1e-8):
    """
    Solve for the eccentric anomaly given the mean anomaly using
    Newton's method

    Parameters
    M: Mean anomaly (rads)
    e: Eccentricity (unitless)
    tol: Tolerance for convergence

    Returns
    E: Eccentric anomaly (rads)
    """
    E = M if e < 0.8 else np.pi  # Initial guess
    for _ in range(100):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if abs(delta) < tol:
            break
    return E


def true_anomaly_from_mean_anomaly(M, e):
    """
    Calculate the true anomaly given the mean anomaly.

    Parameters
    M: Mean anomaly (rads)
    e: Eccentricity (unitless)

    Returns
    true_anomaly: True anomaly (rads)
    """
    E = eccentric_anomaly_from_mean_anomaly(M, e)
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    return true_anomaly


def true_anomaly_at_time(a: float, e: float, M_start, dt: float,
                         mu: float) -> float:
    """
    Calculate the true anomaly of an orbiting body at a given time.

    Parameters
    a: Semi-major axis (m)
    e: Eccentricity (unitless)
    dt: Time since M_start passage (s)
    mu: Standard gravitational parameter (m^3/s^2)

    Returns
    true_anomaly: True anomaly (radians)
    """
    # Mean anomaly update
    n = mean_motion(a, mu)
    mean_anomaly = (M_start + n * dt) % (2 * np.pi)  # Wrap it to 0-2pi

    # True anomaly
    true_anomaly = true_anomaly_from_mean_anomaly(mean_anomaly, e)
    return true_anomaly


def radius_at_true_anomaly(a: float, e: float, theta: float) -> float:
    """
    Calculate the orbital radius at a given true anomaly.

    Parameters
    a: Semi-major axis (m)
    e: Eccentricity (unitless)
    theta: True anomaly (radians)

    Returns
    r: Radius (m)
    """
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    return r


def orbit_shape_from_rv(r, v, mu):
    """
    Calculate the eccentricity of an orbit given radius, velocity, and the
    standard gravitational parameter

    Parameters
    r: Radius from the central body (m)
    v: Orbital velocity at that radius (m/s)
    mu: Gravitational parameter (m^3/s^2)

    Returns
    a: Semi-major axis (m)
    e: Eccentricity (unitless)
    """
    specific_energy = v**2 / 2 - mu / r
    a = -mu / (2 * specific_energy)
    angular_momentum = r * v
    e = np.sqrt(1 - (angular_momentum**2 / (mu * a)))
    return a, e


def separation_distance(position1: NDArray[np.float64],
                        position2: NDArray[np.float64]) -> (
                            NDArray[np.float64]):
    """
    Compute the position difference after one revolution

    Parameters
    position1: nxm matrix of positions (m)
    position2 nxm matrix of positions (m)

    Returns
    separation: separation distances (m)
    """
    separation = np.linalg.norm(position1 - position2)
    return separation


def position_2d(r, theta):
    """
    Compute 2d position given a radius and an angle

    Parameters
    r: Radius (m)
    theta: Angle (rads)

    Returns
    position: 2-Long Vector of x and y positions (m)
    """
    unit_vector = np.array((np.cos(theta), np.sin(theta)))
    position = r * unit_vector
    return position


def separation_after_time(dv, a_initial, e_initial, M_start, dt, mu):
    """
    Compute the separation distance between two satellites at a given time
    after deployment.

    Parameters
    dv: Delta-V (km/s)
    a_initial: Semi-major axis of the initial orbit (m)
    e_initial: Eccentricity of the initial orbit
    M_start: Mean anomaly at separation (rads)
    dt: Time after deployment to check separation (s)
    mu: Standard gravitational parameter (m^3/s^2)

    Returns
    separation: Separation distance (m)
    """
    # Initial true anomaly and orbital radius and velocity
    true_anomaly = true_anomaly_from_mean_anomaly(M_start, e_initial)
    r_deploy = radius_at_true_anomaly(a_initial, e_initial, true_anomaly)
    v_initial = velocity_at_radius(a_initial, r_deploy, mu)

    # Velocity after deployment
    v_deploy = v_initial + dv

    # Compute new orbit's semi-major axis and eccentricity
    a_new, e_new = orbit_shape_from_rv(r_deploy, v_deploy, mu)

    # True anomalies at the time of checking
    theta_initial = true_anomaly_at_time(a_initial, e_initial, M_start, dt, mu)
    theta_new = true_anomaly_at_time(a_new, e_new, M_start, dt, mu)

    # Radii at the time of checking
    r_undeployed = radius_at_true_anomaly(a_initial, e_initial, theta_initial)
    r_deployed = radius_at_true_anomaly(a_new, e_new, theta_new)

    # Separation distance
    initial_position = position_2d(r_undeployed, theta_initial)
    separated_position = position_2d(r_deployed, theta_new)
    separation = separation_distance(initial_position, separated_position)
    return separation


