"""Тесты расчёта кадров для анимаций."""

import numpy as np

from lj_md.animate import simulate_frames


def test_frames_shape():
    """Кадры сохраняются каждые `every` шагов, включая нулевой и последний."""
    data = simulate_frames(N=36, rho=0.3, n_steps=100, every=20, seed=0)
    assert data["frames_pos"].shape == (6, 36, 2)
    assert data["frames_vel"].shape == (6, 36, 2)
    assert list(data["frames_step"]) == [0, 20, 40, 60, 80, 100]


def test_positions_inside_box():
    data = simulate_frames(N=36, rho=0.3, n_steps=200, every=50, seed=0)
    pos = data["frames_pos"]
    assert np.all(pos >= 0) and np.all(pos < data["L"])


def test_cooling_lowers_temperature():
    """При cool_factor < 1 температура падает."""
    data = simulate_frames(N=36, rho=0.5, n_steps=2000, every=500,
                           cool_factor=0.99, cool_every=10, seed=0)
    assert data["frames_T"][-1] < 0.5 * data["frames_T"][0]
