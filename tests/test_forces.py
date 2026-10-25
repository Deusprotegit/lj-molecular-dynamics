"""Тесты модели МД газа Леннарда-Джонса. Запуск из корня проекта: pytest -v"""

import numpy as np

from lj_md.model import (
    compute_forces,
    init_positions,
    init_velocities_maxwell,
    temperature,
    velocity_verlet_step,
)

CUTOFF = 2.5
R2_CUT = CUTOFF**2
U_CUT = 4.0 * (1.0 / R2_CUT**6 - 1.0 / R2_CUT**3)
R_MIN = 2.0 ** (1.0 / 6.0)  # минимум потенциала
L_BIG = 100.0               # большой бокс, чтобы минимальный образ не влиял


def lj_pair_energy(r):
    """Сдвинутый потенциал ЛД для одной пары (эталон для тестов)."""
    return 4.0 * (r**-12 - r**-6) - U_CUT


def test_force_two_particles_r1():
    """Две частицы на расстоянии r=1: |F| = 24, отталкивание."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    forces, _ = compute_forces(pos, L_BIG, cutoff=CUTOFF)
    # Частица 0 отталкивается от 1: F0 направлена в -x
    assert np.isclose(forces[0, 0], -24.0), forces[0, 0]
    assert np.isclose(forces[1, 0], +24.0), forces[1, 0]
    assert np.allclose(forces[:, 1], 0.0)


def test_force_zero_at_rmin():
    """При r = 2^(1/6) сила Леннарда-Джонса равна нулю."""
    pos = np.array([[0.0, 0.0], [R_MIN, 0.0]])
    forces, _ = compute_forces(pos, L_BIG, cutoff=CUTOFF)
    assert np.allclose(forces, 0.0, atol=1e-10), forces


def test_energy_two_particles_r1():
    """U(r=1) = 4*(1 - 1) - U_cut = -U_cut (одна пара, без двойного счёта)."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    _, U = compute_forces(pos, L_BIG, CUTOFF)
    assert np.isclose(U, -U_CUT), U


def test_energy_at_rmin_is_minus_epsilon():
    """В минимуме потенциала U(r_min) = -1 (до сдвига на U_cut)."""
    pos = np.array([[0.0, 0.0], [R_MIN, 0.0]])
    _, U = compute_forces(pos, L_BIG, CUTOFF)
    assert np.isclose(U, -1.0 - U_CUT), U


def test_no_double_counting_N3():
    """Для трёх частиц U — сумма по трём парам (а не шести).

    Две пары на расстоянии r_min (сила 0, но U = -1 у каждой)
    и одна пара на расстоянии r_min*sqrt(2).
    """
    pos = np.array([[0.0, 0.0], [R_MIN, 0.0], [0.0, R_MIN]])
    _, U = compute_forces(pos, L_BIG, CUTOFF)
    expected = 2 * lj_pair_energy(R_MIN) + lj_pair_energy(R_MIN * np.sqrt(2))
    assert np.isclose(U, expected), (U, expected)


def test_force_matches_numerical_derivative():
    """Сила совпадает с численной производной -dU/dx."""
    r, h = 1.3, 1e-6
    _, U_plus = compute_forces(np.array([[0.0, 0.0], [r + h, 0.0]]), L_BIG, CUTOFF)
    _, U_minus = compute_forces(np.array([[0.0, 0.0], [r - h, 0.0]]), L_BIG, CUTOFF)
    F_numeric = -(U_plus - U_minus) / (2 * h)   # сила на частицу 1 вдоль x
    forces, _ = compute_forces(np.array([[0.0, 0.0], [r, 0.0]]), L_BIG, CUTOFF)
    assert np.isclose(forces[1, 0], F_numeric, rtol=1e-6), (forces[1, 0], F_numeric)


def test_newton_third_law():
    """Сумма всех сил в системе равна нулю."""
    rng = np.random.default_rng(0)
    N, rho = 100, 0.3
    L = np.sqrt(N / rho)
    pos = init_positions(N, L, rng)
    forces, _ = compute_forces(pos, L, CUTOFF)
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-10)


def test_minimum_image():
    """Частицы у противоположных краёв ячейки взаимодействуют через границу."""
    L = 10.0
    pos_edge = np.array([[0.1, 5.0], [L - 0.9, 5.0]])   # через границу r = 1
    pos_inside = np.array([[5.0, 5.0], [6.0, 5.0]])      # обычные r = 1
    F_edge, U_edge = compute_forces(pos_edge, L, CUTOFF)
    F_inside, U_inside = compute_forces(pos_inside, L, CUTOFF)
    assert np.isclose(U_edge, U_inside)
    # Частица 0 отталкивается от соседа слева (через границу) -> F0 в +x
    assert np.isclose(F_edge[0, 0], -F_inside[0, 0])


def test_temperature_definition():
    """T = K/(N-1) в 2D после удаления импульса."""
    rng = np.random.default_rng(0)
    vel = init_velocities_maxwell(100, T_target=1.5, rng=rng)
    assert np.isclose(temperature(vel), 1.5, atol=1e-10)


def test_momentum_removed():
    rng = np.random.default_rng(1)
    vel = init_velocities_maxwell(50, 1.0, rng)
    assert np.allclose(vel.sum(axis=0), 0.0, atol=1e-12)


def test_momentum_conserved_verlet():
    """Суммарный импульс остаётся нулевым после шагов Верле."""
    rng = np.random.default_rng(2)
    N, rho, dt = 64, 0.3, 0.005
    L = np.sqrt(N / rho)
    pos = init_positions(N, L, rng)
    vel = init_velocities_maxwell(N, 1.0, rng)
    forces, _ = compute_forces(pos, L, CUTOFF)
    for _ in range(100):
        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, dt, L, CUTOFF)
    assert np.allclose(vel.sum(axis=0), 0.0, atol=1e-10)
