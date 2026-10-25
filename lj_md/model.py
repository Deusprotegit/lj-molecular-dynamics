import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# ============================================================
# Параметры системы (приведённые единицы: m = sigma = epsilon = 1)
# ============================================================
N = 200                 # число частиц
rho = 0.3               # плотность (N / L^2)
L = np.sqrt(N / rho)    # размер квадратной ячейки
T_target = 1.0          # целевая температура
dt = 0.005              # шаг по времени
N_steps = 20000         # полное число шагов
N_equil = 5000          # шагов на термализацию
N_sample = 50           # частота сбора статистики

cutoff = 2.5            # радиус обрезания потенциала
r2_cut = cutoff**2

np.random.seed(42)

# ============================================================
# Начальные координаты (решётка) и скорости (Максвелл)
# ============================================================
def init_positions(N, L):
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            if len(pos) < N:
                pos.append([(i + 0.5) * spacing, (j + 0.5) * spacing])
    return np.array(pos)

pos = init_positions(N, L)
# небольшое случайное смещение, чтобы избежать идеальной симметрии
pos += np.random.uniform(-0.05, 0.05, pos.shape)

vel = np.random.normal(0.0, np.sqrt(T_target), size=(N, 2))

# Убираем суммарный импульс и задаём точную температуру
vel -= vel.mean(axis=0)
# В 2D: <v^2>/2 = T  =>  кинетическая энергия = N*T
KE = 0.5 * np.sum(vel**2)
vel *= np.sqrt(N * T_target / KE)  # теперь <v^2> = 2*T

# ============================================================
# Силы Леннарда-Джонса (2D)
# ============================================================
def compute_forces(pos, L):
    """Возвращает силы и потенциальную энергию (минимальный образ)."""
    N = len(pos)
    forces = np.zeros_like(pos)
    U = 0.0
    # Векторизованный расчёт через broadcasting
    d = pos[:, None, :] - pos[None, :, :]   # (N, N, 2)
    # Минимальный образ (периодические границы)
    d -= L * np.round(d / L)
    r2 = np.sum(d * d, axis=-1)

    # Маскируем самодействие и r > cutoff
    np.fill_diagonal(r2, np.inf)
    mask = r2 < r2_cut

    r2_m = r2[mask]
    inv_r2 = 1.0 / r2_m
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2

    # U = 4*(r^-12 - r^-6)  - U_cut
    U_cut = 4.0 * (1.0/r2_cut**6 - 1.0/r2_cut**3)
    U = np.sum(4.0 * (inv_r12 - inv_r6) - U_cut)

    # F_ij = 24*(2*r^-14 - r^-8) * r_vec
    # или: F = 24*(2*inv_r12 - inv_r6)/r2 * r_vec
    coef = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2  # скаляр на пару

    # Вектор силы, действующей на i со стороны j: F_ij = coef * d_ij
    # d_ij = r_i - r_j,  сила притяжения направлена к j (знак отриц.)
    # Стандартно: F_i = sum_j coef * d_ij  (при coef>0 отталкивание при малых r)
    d_full = d[mask]  # (M, 2)
    # Нужно правильно разложить по частицам
    idx_i, idx_j = np.where(mask)
    f_contrib = coef[:, None] * d_full   # (M, 2)
    np.add.at(forces, idx_i, f_contrib)
    np.add.at(forces, idx_j, -f_contrib)

    return forces, U


def kinetic_energy(vel):
    return 0.5 * np.sum(vel**2)


# ============================================================
# Интегрирование методом Верле (velocity Verlet)
# ============================================================
def velocity_verlet_step(pos, vel, forces, dt, L):
    vel += 0.5 * dt * forces
    pos += dt * vel
    pos %= L  # периодические границы
    forces_new, U = compute_forces(pos, L)
    vel += 0.5 * dt * forces_new
    return pos, vel, forces_new, U


# ============================================================
# Основной цикл
# ============================================================
forces, U = compute_forces(pos, L)

# Для сбора статистики скоростей
v_samples = []
T_hist = []
E_hist = []

for step in range(N_steps):
    pos, vel, forces, U = velocity_verlet_step(pos, vel, forces, dt, L)

    if step >= N_equil:
        T_inst = np.sum(vel**2) / (2 * N)   # в 2D: <v^2> = 2T
        T_hist.append(T_inst)
        E_hist.append(kinetic_energy(vel) + U)

        if (step - N_equil) % N_sample == 0:
            v_samples.append(vel.copy())

# Собираем все скорости в один массив
V = np.vstack(v_samples)               # (M, 2)
speed = np.sqrt(np.sum(V**2, axis=1))  # модуль скорости

# ============================================================
# Проверка распределения Максвелла
# ============================================================
# В 2D распределение модуля скорости:
#   f(v) = (v / T) * exp(-v^2 / (2T))
# Проверим соответствие для каждой компоненты (нормальное) и для модуля.

T_measured = np.mean(T_hist)
vx = V[:, 0]
vy = V[:, 1]

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Компонента vx (нормальное распределение N(0, T)) ---
axs[0].hist(vx, bins=60, density=True, alpha=0.7, color='steelblue', label='MD')
v_grid = np.linspace(-4, 4, 200)
f_norm = 1/np.sqrt(2*np.pi*T_measured) * np.exp(-v_grid**2/(2*T_measured))
axs[0].plot(v_grid, f_norm, 'r-', lw=2, label=f'N(0, T={T_measured:.3f})')
axs[0].set_xlabel('v_x'); axs[0].set_ylabel('P(v_x)')
axs[0].set_title('Компонента скорости')
axs[0].legend()

# --- Модуль скорости: f(v) = v/T * exp(-v²/2T) ---
axs[1].hist(speed, bins=60, density=True, alpha=0.7, color='seagreen', label='MD')
v_grid = np.linspace(0, speed.max(), 300)
f_speed = (v_grid / T_measured) * np.exp(-v_grid**2 / (2*T_measured))
axs[1].plot(v_grid, f_speed, 'r-', lw=2, label=f'v/T·e^(-v²/2T), T={T_measured:.3f}')
axs[1].set_xlabel('|v|'); axs[1].set_ylabel('f(|v|)')
axs[1].set_title('Распределение модуля скорости')
axs[1].legend()

# --- Распределение полной энергии по шагам (проверка сохранения) ---
axs[2].plot(E_hist, lw=0.8)
axs[2].set_xlabel('шаг'); axs[2].set_ylabel('E = K + U')
axs[2].set_title('Полная энергия')
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('md_lj_2d_maxwell.png', dpi=120)
plt.show()

# ============================================================
# Количественные проверки
# ============================================================
print(f"Измеренная температура (2D): T = {T_measured:.4f} (цель {T_target})")
print(f"<v_x²> = {np.mean(vx**2):.4f}, ожидание T = {T_measured:.4f}")
print(f"<v_y²> = {np.mean(vy**2):.4f}, ожидание T = {T_measured:.4f}")
print(f"<v²>   = {np.mean(speed**2):.4f}, ожидание 2T = {2*T_measured:.4f}")
print(f"Среднее полной энергии: {np.mean(E_hist):.4f} ± {np.std(E_hist):.4f}")

# Критерий Колмогорова–Смирнова против теоретического максвелла
from scipy.stats import kstest
cdf = lambda v: 1 - np.exp(-v**2 / (2*T_measured))  # CDF для 2D Максвелла
ks = kstest(speed, cdf)
print(f"KS-тест модуля скорости: статистика={ks.statistic:.4f}, p-value={ks.pvalue:.4f}")