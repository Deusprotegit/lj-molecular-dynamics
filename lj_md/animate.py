"""
Анимации МД-модели двумерного газа Леннарда-Джонса.

Расчёт и отрисовка разделены: сначала simulate_frames() считает траекторию
и сохраняет кадры (координаты + скорости), затем функции make_*_gif()
собирают из кадров GIF через celluloid.

Запуск из корня проекта:
    python -m lj_md.animate --what particles   # движение частиц, цвет = |v|
    python -m lj_md.animate --what relax       # релаксация к распределению Рэлея
    python -m lj_md.animate --what cool        # охлаждение -> кристаллизация
    python -m lj_md.animate --what all         # всё сразу

GIF сохраняются в папку results/.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from celluloid import Camera

from lj_md.model import (
    _results_path,
    compute_forces,
    init_positions,
    init_velocities_equal_speed,
    init_velocities_maxwell,
    temperature,
    velocity_verlet_step,
)

CMAP = "plasma"


# ============================================================
# Расчёт кадров
# ============================================================
def simulate_frames(N=200, rho=0.3, T_target=1.0, dt=0.005, n_steps=4000,
                    every=25, init_mode="maxwell", cool_factor=None,
                    cool_every=10, cutoff=2.5, seed=42):
    """
    Прогон МД с сохранением кадров каждые `every` шагов.

    cool_factor — если задан (например, 0.999), то каждые `cool_every` шагов
    скорости домножаются на этот множитель (медленное охлаждение).

    Возвращает dict: L, dt, every, frames_pos (F, N, 2), frames_vel (F, N, 2),
    frames_step (F,), frames_T (F,).
    """
    rng = np.random.default_rng(seed)
    L = np.sqrt(N / rho)
    pos = init_positions(N, L, rng)
    if init_mode == "maxwell":
        vel = init_velocities_maxwell(N, T_target, rng)
    elif init_mode == "equal_speed":
        vel = init_velocities_equal_speed(N, T_target, rng)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")

    forces, _ = compute_forces(pos, L, cutoff)
    fp, fv, fs, fT = [], [], [], []

    for step in range(n_steps + 1):
        if step % every == 0:
            fp.append(pos.copy())
            fv.append(vel.copy())
            fs.append(step)
            fT.append(temperature(vel))
        if step == n_steps:
            break
        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, dt, L, cutoff)
        if cool_factor is not None and (step + 1) % cool_every == 0:
            vel *= cool_factor

    return dict(L=L, dt=dt, every=every,
                frames_pos=np.array(fp), frames_vel=np.array(fv),
                frames_step=np.array(fs), frames_T=np.array(fT))


# ============================================================
# Вспомогательное
# ============================================================
def _marker_size(ax, fig, L, diameter=1.0):
    """Площадь маркера scatter (pt^2), чтобы частица имела диаметр ~ sigma."""
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    pt_per_unit = bbox.width * 72.0 / L
    return (diameter * pt_per_unit) ** 2


def _setup_box(ax, L):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _rayleigh(v, T):
    """Распределение модуля скорости в 2D (m = k_B = 1)."""
    return (v / T) * np.exp(-v**2 / (2 * T))


# ============================================================
# Анимация 1: движение частиц
# ============================================================
def make_particles_gif(data, outfile=None, vmax=None, fps=15, dpi=72,
                       title="Газ Леннарда-Джонса"):
    """Частицы в ячейке, цвет — модуль скорости (шкала одинакова для всех кадров)."""
    outfile = outfile or _results_path("md_particles.gif")
    L, dt = data["L"], data["dt"]
    speeds = np.linalg.norm(data["frames_vel"], axis=-1)
    vmax = vmax or float(np.percentile(speeds, 99))

    fig, ax = plt.subplots(figsize=(5.2, 5.9))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.87)
    _setup_box(ax, L)
    fig.suptitle(title, fontsize=13)
    s = _marker_size(ax, fig, L)
    camera = Camera(fig)

    sc = None
    for pos, sp, step, T in zip(data["frames_pos"], speeds,
                                data["frames_step"], data["frames_T"]):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=sp, s=s, cmap=CMAP,
                        vmin=0.0, vmax=vmax, edgecolors="none")
        # подпись над ячейкой, чтобы частицы её не перекрывали
        ax.text(0.5, 1.01, f"t = {step * dt:6.2f} τ      T = {T:.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        camera.snap()

    cax = fig.add_axes([0.15, 0.08, 0.7, 0.025])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label="|v|")
    camera.animate(interval=1000 / fps).save(outfile, writer="pillow", dpi=dpi)
    plt.close(fig)
    print(f"[make_particles_gif] сохранено: {outfile}")
    return outfile


# ============================================================
# Анимация 2: релаксация распределения скоростей
# ============================================================
def make_relaxation_gif(data, outfile=None, fps=12, dpi=70, bins=30):
    """Слева частицы, справа гистограмма |v| и кривая Рэлея при текущей T."""
    outfile = outfile or _results_path("md_relaxation.gif")
    L, dt = data["L"], data["dt"]
    speeds = np.linalg.norm(data["frames_vel"], axis=-1)
    vmax_c = float(np.percentile(speeds[len(speeds) // 2:], 99))
    v_hi = float(np.percentile(speeds, 99.9)) * 1.1
    edges = np.linspace(0, v_hi, bins + 1)
    v_grid = np.linspace(0, v_hi, 300)
    y_max = 1.2 * float(np.max(_rayleigh(v_grid, data["frames_T"][-1])))

    fig, (ax_p, ax_h) = plt.subplots(1, 2, figsize=(10, 4.8),
                                     gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.13, top=0.88, wspace=0.12)
    _setup_box(ax_p, L)
    ax_p.set_title("Частицы (цвет = |v|)")
    ax_h.set_xlim(0, v_hi)
    ax_h.set_ylim(0, y_max)
    ax_h.set_xlabel("|v|")
    ax_h.set_ylabel("f(|v|)")
    ax_h.set_title("Распределение модуля скорости")
    ax_h.grid(alpha=0.3)
    s = _marker_size(ax_p, fig, L)
    camera = Camera(fig)

    for pos, sp, step, T in zip(data["frames_pos"], speeds,
                                data["frames_step"], data["frames_T"]):
        ax_p.scatter(pos[:, 0], pos[:, 1], c=sp, s=s, cmap=CMAP,
                     vmin=0.0, vmax=vmax_c, edgecolors="none")
        ax_h.hist(sp, bins=edges, density=True, color="steelblue", alpha=0.75)
        ax_h.plot(v_grid, _rayleigh(v_grid, T), "r-", lw=2)
        ax_h.text(0.97, 0.95,
                  f"t = {step * dt:5.2f} τ\nT = {T:.2f}\n— Рэлей: (v/T)·exp(−v²/2T)",
                  transform=ax_h.transAxes, ha="right", va="top", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        camera.snap()

    camera.animate(interval=1000 / fps).save(outfile, writer="pillow", dpi=dpi)
    plt.close(fig)
    print(f"[make_relaxation_gif] сохранено: {outfile}")
    return outfile


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Анимации 2D LJ MD")
    p.add_argument("--what", choices=["particles", "relax", "cool", "all"],
                   default="all")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.what in ("particles", "all"):
        data = simulate_frames(N=args.n, rho=0.3, T_target=1.0,
                               n_steps=4000, every=40, seed=args.seed)
        make_particles_gif(data)

    if args.what in ("relax", "all"):
        # Старт со всех одинаковых |v|: распределение «расплывается»
        # и приходит к Рэлею. Сначала кадры чаще — там самое интересное.
        data = simulate_frames(N=args.n, rho=0.3, T_target=1.0,
                               n_steps=1500, every=15,
                               init_mode="equal_speed", seed=args.seed)
        make_relaxation_gif(data)

    if args.what in ("cool", "all"):
        # Плотная система, медленное охлаждение -> гексагональная упаковка
        data = simulate_frames(N=args.n, rho=0.75, T_target=1.0,
                               n_steps=16000, every=160,
                               cool_factor=0.999, cool_every=10,
                               seed=args.seed)
        make_particles_gif(data, outfile=_results_path("md_cooling.gif"),
                           vmax=2.0, fps=12,
                           title="Охлаждение → кристаллизация")


if __name__ == "__main__":
    main()
