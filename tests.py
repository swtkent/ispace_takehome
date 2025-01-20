import numpy as np
import lunar_transfer_utils as ltu


def test_orbital_velocity():
    a = 7000  # Semi-major axis (km)
    r = 7100  # Orbital radius (km)
    v = ltu.velocity_at_radius(a, r, ltu.MU_MOON)
    assert v > 0, "Orbital velocity should be positive."


def test_eccentric_anomaly_from_mean_anomaly():
    M = np.pi / 2  # Mean anomaly (rads)
    e = 0.1  # Eccentricity
    E = ltu.eccentric_anomaly_from_mean_anomaly(M, e)
    assert 0 <= E <= 2 * np.pi, "Eccentric anomaly should be between 0 and 2*pi."


def test_true_anomaly_from_mean_anomaly():
    M = np.pi / 2  # Mean anomaly (rads)
    e = 0.1  # Eccentricity
    theta = ltu.true_anomaly_from_mean_anomaly(M, e)
    assert -np.pi <= theta <= np.pi, "True anomaly should be between -pi and pi."


def test_orbit_shape_from_rv():
    r = 7100  # Radius (km)
    v = 7.12  # Velocity (km/s)
    a, e = ltu.orbit_shape_from_rv(r, v, ltu.MU_MOON)
    assert a > 0
    assert 0 <= e < 1, "Eccentricity should be between 0 and 1 for bounded orbits."


def test_true_anomaly_at_time():
    a = 7000  # Semi-major axis (km)
    e = 0.1  # Eccentricity
    t = 3600  # Time (s)
    theta = ltu.true_anomaly_at_time(a, e, 0, t, ltu.MU_MOON)
    assert -np.pi <= theta <= np.pi, "True anomaly should be between -pi and pi."


def test_radius_at_true_anomaly():
    a = 7000  # Semi-major axis (km)
    e = 0.1  # Eccentricity
    theta = np.pi / 3  # True anomaly (rads)
    r = ltu.radius_at_true_anomaly(a, e, theta)
    assert r > 0, "Radius should be positive."


def test_separation_after_time():
    dv = 0.01  # Delta-V (km/s)
    a_initial = 7000  # Semi-major axis (km)
    e_initial = 0.1  # Eccentricity
    t_check = 3600  # Time (s)
    separation = ltu.separation_after_time(dv, a_initial, e_initial, 0,
                                           t_check, ltu.MU_MOON)
    assert separation > 0, "Separation should be positive."
