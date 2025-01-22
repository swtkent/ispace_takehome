import pytest as pt
import numpy as np
import lunar_transfer_utils as ltu


def test_orbit_shape_from_altitudes():
    radius = 100  # m
    h_p = 100  # m
    h_a = 500  # m
    a, e = ltu.orbit_shape_from_altitudes(h_p, h_a, radius)
    expected_a = 400
    expected_e = 0.50
    assert a > 0, "Semi-major axis should be positive for bounded orbits."
    assert 0 <= e < 1, "Eccentricity should be between 0 and 1 for bounded orbits."
    assert a == pt.approx(expected_a, abs=1e0), f"Expected semi-major axis value of {expected_a:.0f} m instead of {a:.0f}"
    assert e == pt.approx(expected_e, abs=1e-2), f"Expected eccentricity of {expected_e:.2f} instead of {e:.2f}"


def test_mean_motion():
    a = 7300e3  # m
    mu = ltu.MU_MOON  # m^3/s^2
    n = ltu.mean_motion(a, mu)
    expected = 0.00011
    assert n > 0, "Mean motion should be positive for bounded orbits."
    assert n == pt.approx(expected, abs=1e-5), f"Expected mean motion of {expected:.5f} rad/s instead of {n:.5f}"

def test_period_from_mm():
    n = np.pi / 4 # rad/s
    P = ltu.period_from_mm(n)
    expected = 8.0
    assert P > 0, "Period should be positive for bounded orbits."
    assert P == pt.approx(expected, abs=1e-3), f"Expected period of {expected:.3f} s instead of {P:.3f}"


def test_period_from_sma():
    a = 7000e3  # m
    P = ltu.period_from_sma(a, ltu.MU_MOON)
    expected = 52553.726
    assert P > 0, "Period should be positive for bounded orbits."
    assert P == pt.approx(expected, abs=1e-3), f"Expected period of {expected:.3f} s instead of {P:.3f}"



def test_velocity_at_radius():
    a = 7000e3  # Semi-major axis (m)
    r = 7100e3  # Orbital radius (m)
    v = ltu.velocity_at_radius(a, r, ltu.MU_MOON)
    expected = 825.030
    assert v > 0, "Orbital velocity should be positive."
    assert v == pt.approx(expected, abs=1e-3), f"Expected velocity value of {expected:.3f} m/s instead of {v:.3f}"


def test_eccentric_anomaly_from_mean_anomaly():
    M = np.pi / 2  # Mean anomaly (rads)
    e = 0.1  # Eccentricity
    E = ltu.eccentric_anomaly_from_mean_anomaly(M, e)
    expected = 1.670
    assert 0 <= E <= 2 * np.pi, "Eccentric anomaly should be between 0 and 2*pi."
    assert E == pt.approx(expected, abs=1e-3), f"Expected eccentric anomaly value of {expected:.3f} rads instead of {E:.3f}"


def test_true_anomaly_from_eccentric_anomaly():
    E = 1.670  # Eccentric anomaly (rads)
    e = 0.1  # Eccentricity
    theta = ltu.true_anomaly_from_eccentric_anomaly(E, e)
    expected = 1.769
    assert -np.pi <= theta <= np.pi, "True anomaly should be between -pi and pi."
    assert theta == pt.approx(expected, abs=1e-3), f"Expected true anomaly value of {expected:.3f} rads instead of {theta:.3f}"


def test_true_anomaly_from_mean_anomaly():
    M = np.pi / 2  # Mean anomaly (rads)
    e = 0.1  # Eccentricity
    theta = ltu.true_anomaly_from_mean_anomaly(M, e)
    expected = 1.769
    assert -np.pi <= theta <= np.pi, "True anomaly should be between -pi and pi."
    assert theta == pt.approx(expected, abs=1e-3), f"Expected true anomaly value of {expected:.3f} rads instead of {theta:.3f}"


def test_true_anomaly_at_time():
    a = 7000e3  # Semi-major axis (m)
    e = 0.1  # Eccentricity
    t = 3600  # Time (s)
    theta = ltu.true_anomaly_at_time(a, e, 0, t, ltu.MU_MOON)
    expected = 0.524
    assert -np.pi <= theta <= np.pi, "True anomaly should be between -pi and pi."
    assert theta == pt.approx(expected, abs=1e-3), f"Expected true anomaly of {expected:.3f} rads instead of {theta:.3f}"


def test_radius_at_true_anomaly():
    a = 7000  # Semi-major axis (m)
    e = 0.1  # Eccentricity
    theta = np.pi / 3  # True anomaly (rads)
    r = ltu.radius_at_true_anomaly(a, e, theta)
    expected = 6600.00
    assert r > 0, "Radius should be positive."
    assert r == pt.approx(expected, 1e-2), f"Expected radius of {expected:.2f} m instead of {r:.2f}"


def test_orbit_shape_from_rv():
    r = 7100e3  # Radius (m)
    v = 825.03  # Velocity (m/s)
    a, e = ltu.orbit_shape_from_rv(r, v, ltu.MU_MOON)
    expected_a = 7000e3
    expected_e = 0.01429
    assert a > 0, "Semi-major axis should be positive for bounded orbits."
    assert 0 <= e < 1, "Eccentricity should be between 0 and 1 for bounded orbits."
    assert a == pt.approx(expected_a, abs=1e0), f"Expected semi-major axis value of {expected_a:.1f} m instead of {a:.1f}"
    assert e == pt.approx(expected_e, abs=1e-4), f"Expected eccentricity of {expected_e:.4f} instead of {e:.4f}"


def test_separation_after_time():
    dv = 10  # Delta-V (m/s)
    a_initial = 7000e3  # Semi-major axis (m)
    e_initial = 0.1  # Eccentricity
    t_check = 3600  # Time (s)
    separation = ltu.separation_after_time(dv, a_initial, e_initial, 0,
                                            t_check, ltu.MU_MOON)
    expected = 34892.49
    assert separation > 0, "Separation should be positive."
    assert separation == pt.approx(expected, 1e-2), f"Expected separation of {expected:.2f} m instead of {separation:.2f}"