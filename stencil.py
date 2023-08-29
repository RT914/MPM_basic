import numpy as np
import taichi as ti
from numpy import linalg
import random as ra
import matplotlib.cm as cm

# 疑似乱数のシード値
ra.seed(312)

ti.init(arch=ti.gpu)

# Window Size
n = 500
gui = ti.GUI("MPM", res=(n, n), background_color=0xffffff, fullscreen=False)

n_particles: int = 3000
n_grid = 30
n_offset = 16
dimensions = 2

difference = 1 / n_grid
dt = 5.0e-4
limit_v = 0.01 / dt  # 制限速度

# 初期設定
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # 位置座標
mass = ti.field(dtype=float, shape=n_particles)  # 質量
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # 速度
grid_mass = ti.field(
    dtype=float, shape=(n_grid, n_grid)
)
grid_momentum = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)
grid_location = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)
grid_location_copy = ti.Vector.field(
    2, dtype=float, shape=n_grid * n_grid
)
grid_v = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)
p_volume = ti.field(
    dtype=float, shape=n_particles
)
gravity = ti.Vector.field(
    2, dtype=float, shape=1
)
gravity[0] = [0.0, -9]
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())
F = ti.Matrix.field(
    2, 2, dtype=float, shape=n_particles
)
F_copy = ti.Matrix.field(
    2, 2, dtype=float, shape=n_particles
)
J = ti.field(
    dtype=float, shape=n_particles
)  # 体積変化率
J_copy = ti.field(
    dtype=float, shape=n_particles
)
E = 140  # ヤング率
poisson_ratio = 0.2  # ポイソン比率
micro = E / (2 * (1 + poisson_ratio))  # μ
lamda = (E * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))  # λ
sigma = ti.Matrix.field(
    2, 2, dtype=float, shape=n_particles
)  # 応力テンソル
f = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)  # 応力
weight_vector = ti.Vector.field(
    2, dtype=float, shape=n_particles
)
f_divide_m = ti.Vector.field(
    2, dtype=float, shape=1
)
sigma_x = ti.Vector.field(
    2, dtype=float, shape=n_particles
)
sigma_y = ti.Vector.field(
    2, dtype=float, shape=n_particles
)
f_grid_copy = ti.Vector.field(
    2, dtype=float, shape=n_grid * n_grid
)
vector_lines = ti.Vector.field(
    2, dtype=float, shape=n_grid * n_grid
)
grid_v_copy = ti.Vector.field(
    2, dtype=float, shape=n_grid * n_grid
)
grid_mass_copy = ti.field(
    dtype=float, shape=n_grid * n_grid
)
offset = ti.Vector.field(
    2, dtype=int, shape=(n_particles, n_offset)
)
offset_position = ti.Vector.field(
    2, dtype=float, shape=n_particles
)
weight = ti.field(
    dtype=float, shape=(n_particles, n_offset)
)
derivative_weight = ti.Vector.field(
    2, dtype=float, shape=(n_particles, n_offset)
)

for a in range(n_particles):
    x[a] = [
        ra.uniform(0.4, 0.6), ra.uniform(0.4, 0.6)
        # 0.4, 0.4
    ]
    v[a] = [0.0, 0.0]
    mass[a] = 1.0
    F[a] = [[1, 0], [0, 1]]
    J[a] = np.linalg.det(F[a])
    sigma[a] = [[1, 0], [0, 1]]


# 格子線
begin = []
end = []
# 縦
for a in range(n_grid):
    begin.append(1.0 / n_grid * a)
    begin.append(0.0)
    end.append(1.0 / n_grid * a)
    end.append(1.0)

# 横
for a in range(n_grid):
    begin.append(0.0)
    begin.append(1.0 / n_grid * a)
    end.append(1.0)
    end.append(1.0 / n_grid * a)

x_t = np.reshape(begin, (n_grid * 2, 2))
y_t = np.reshape(end, (n_grid * 2, 2))

# 格子点の位置設定
grid_num = 0
for a in range(n_grid):
    for b in range(n_grid):
        grid_location[a, b] = [a / n_grid, b / n_grid]
        grid_location_copy[a * n_grid + b] = [-1.0, -1.0]


# 四捨五入
@ti.func
def Round(matrix, num):
    m = ti.round(matrix * pow(10, (num - 1.0)))
    return m / pow(10, (num - 1.0))


# grid statusの初期化
@ti.func
def initialization_grid(num1, num2):
    grid_v[num1, num2] = [0.0, 0.0]
    grid_mass[num1, num2] = 0.0
    grid_momentum[num1, num2] = [0.0, 0.0]
    f[num1, num2] = [0.0, 0.0]
    grid_location_copy[num1 * n_grid + num2] = [-1.0, -1.0]


@ti.func
def initialization_offset(num1, num2):
    offset_position[num1] = [0.0, 0.0]
    offset[num1, num2] = [0, 0]


# 重みづけ関数その1
@ti.func
def Weighting(dist: float) -> float:
    dist = ti.abs(dist)
    kernel_value = 0.0
    if dist >= 2.0:
        kernel_value = 0.0
    elif (dist >= 0.0) and (dist < 1.0):
        kernel_value = 0.5 * pow(dist, 3) - pow(dist, 2) + (2 / 3)
    elif (dist >= 1) and (dist < 2):
        kernel_value = pow((2 - dist), 3) * (1 / 6)
    return kernel_value


# 重みづけ関数その2
@ti.func
def Weighting_dash(dist: float) -> float:
    sgn = 1.0 if dist >= 0.0 else -1.0
    dist = ti.abs(dist)
    kernel_value = 0.0
    if dist >= 2.0:
        kernel_value = 0.0
    elif (dist >= 0.0) and (dist < 1.0):
        kernel_value = 1.5 * pow(dist, 2) - 2 * dist
    elif (dist >= 1.0) and (dist < 2.0):
        kernel_value = 0.5 * (- pow(dist, 2) + 4 * dist - 4)
    return kernel_value * sgn


@ti.func
# offsetの設定
def set_offset(base_offset):
    base_x = offset[base_offset, 0][0]
    base_y = offset[base_offset, 0][1]
    offset_num = 0
    for y in range(n_offset - 1):
        offset_num = y + 1
        if (offset_num == 4) | (offset_num == 8) | (offset_num == 12):
            base_y += 1

        offset[base_offset, offset_num] = [base_x + offset_num % 4, base_y]


@ti.kernel
def decision_offset():
    for i1 in range(n_particles):
        # offsetの初期化
        for i4 in range(n_offset):
            initialization_offset(i1, i4)

        for i2 in range(n_grid):
            for i3 in range(n_grid):
                # 0番目のoffsetの判別
                if x[i1][0] >= grid_location[i2, i3][0]:
                    if grid_location[i2, i3][0] > offset_position[i1][0]:
                        offset[i1, 0][0] = i2
                        offset_position[i1][0] = grid_location[i2, i3][0]
                if x[i1][1] >= grid_location[i2, i3][1]:
                    if grid_location[i2, i3][1] > offset_position[i1][1]:
                        offset[i1, 0][1] = i3
                        offset_position[i1][1] = grid_location[i2, i3][1]

        # 基準offsetの設定
        offset[i1, 0] = [offset[i1, 0][0] - 1, offset[i1, 0][1] - 1]
        set_offset(i1)

        weight_check = 0.0

        for i4 in range(n_offset):
            offset_x = offset[i1, i4][0]
            offset_y = offset[i1, i4][1]
            # 重み
            weight[i1, i4] = \
                Weighting((x[i1][0] - grid_location[offset_x, offset_y][0]) / difference) * \
                Weighting((x[i1][1] - grid_location[offset_x, offset_y][1]) / difference)
            weight_check += weight[i1, i4]

            derivative_weight_x = Weighting_dash((x[i1][0] - grid_location[offset_x, offset_y][0]) / difference) * \
                Weighting((x[i1][1] - grid_location[offset_x, offset_y][1]) / difference) / difference
            derivative_weight_y = Weighting((x[i1][0] - grid_location[offset_x, offset_y][0]) / difference) * \
                Weighting_dash((x[i1][1] - grid_location[offset_x, offset_y][1]) / difference) / difference
            derivative_weight[i1, i4] = [derivative_weight_x, derivative_weight_y]
            # print(derivative_weight[i1, i4])

        # print(weight_check)


# 体積推定
@ti.kernel
def estimate_Volume():
    for i in range(n_particles):
        num = 0.0
        for j in range(n_offset):
            offset_x = offset[i, j][0]
            offset_y = offset[i, j][1]
            # 体積推定
            num += grid_mass[offset_x, offset_y] * weight[i, j] * difference**2
        if num > 0.0:
            p_volume[i] = mass[i] / num


# Particle to Grid
@ti.kernel
def P2G():
    for i in range(n_grid):
        for j in range(n_grid):
            # grid statusの初期化
            initialization_grid(i, j)

    for i in range(n_particles):
        for j in range(n_offset):
            offset_x = offset[i, j][0]
            offset_y = offset[i, j][1]
            # 運動量情報を粒子から格子点へ
            grid_momentum[offset_x, offset_y] += mass[i] * v[i] * weight[i, j]
            # 質量情報を粒子から格子点へ
            grid_mass[offset_x, offset_y] += mass[i] * weight[i, j]

            if grid_mass[offset_x, offset_y] > 0:
                grid_v[offset_x, offset_y] = grid_momentum[offset_x, offset_y] / grid_mass[offset_x, offset_y]
                grid_mass_copy[n_grid * offset_x + offset_y] = grid_mass[offset_x, offset_y]


# 応力テンソル計算
@ti.kernel
def compute_stress():
    for i in range(n_particles):
        #  Piola-Kirchhoff
        piola = micro * (F[i] - F[i].transpose().inverse()) + lamda * ti.log(J[i]) * F[i].transpose().inverse()
        # 応力
        sigma[i] = (1 / J[i]) * (piola @ F[i].transpose())

        # sigmaの描画のための要素
        sigma_x[i][0] = sigma[i][0, 0]/100
        sigma_x[i][1] = sigma[i][1, 0]/100
        sigma_x[i] += x[i]
        sigma_y[i][0] = sigma[i][0, 1]/100
        sigma_y[i][1] = sigma[i][1, 1]/100
        sigma_y[i] += x[i]


# 内力による速度更新
@ti.kernel
def update_velocity():
    # 内力計算
    for i in range(n_particles):
        for j in range(n_offset):
            offset_x = offset[i, j][0]
            offset_y = offset[i, j][1]
            # print(f[offset_x, offset_y])
            # print("p_volume", p_volume[i])
            f[offset_x, offset_y][0] += \
                p_volume[i] * J[i] * \
                (sigma[i][0, 0] * derivative_weight[i, j][0] + sigma[i][0, 1] * derivative_weight[i, j][1]) * (-1)

            f[offset_x, offset_y][1] += \
                p_volume[i] * J[i] * \
                (sigma[i][1, 0] * derivative_weight[i, j][0] + sigma[i][1, 1] * derivative_weight[i, j][1]) * (-1)
            f_divide_m_0 = 0.0
            f_divide_m_1 = 0.0

            f_grid_copy[n_grid*offset_x+offset_y] \
                = f[offset_x, offset_y]/100 + grid_location_copy[n_grid*offset_x+offset_y]
            # f/m
            if f[offset_x, offset_y][0] == 0.0 or grid_mass[offset_x, offset_y] <= 1.0e-5:
                f_divide_m_0 = 0.0
            else:
                f_divide_m_0 = f[offset_x, offset_y][0] / grid_mass[offset_x, offset_y]

            if f[offset_x, offset_y][1] == 0.0 or grid_mass[offset_x, offset_y] <= 1.0e-5:
                f_divide_m_1 = 0.0
            else:
                f_divide_m_1 = f[offset_x, offset_y][1] / grid_mass[offset_x, offset_y]

            f_divide_m[0] = [f_divide_m_0, f_divide_m_1]
            # print("f", f_divide_m[0] * 1000)
            # 格子点の速度更新
            grid_v[offset_x, offset_y] = grid_v[offset_x, offset_y] + dt * f_divide_m[0]
            # print(grid_v[offset_x, offset_y])

            # 格子点速度のコピー
            grid_v_copy[n_grid*offset_x+offset_y] = grid_v[offset_x, offset_y]
            vector_lines[n_grid*offset_x+offset_y] \
                = grid_location_copy[n_grid*offset_x+offset_y] + grid_v_copy[n_grid*offset_x+offset_y]


# 変形勾配の更新
@ti.kernel
def update_Deformation_gradient():
    for i in range(n_particles):
        ft = ti.Matrix.identity(float, 2)
        for j in range(n_offset):
            offset_x = offset[i, j][0]
            offset_y = offset[i, j][1]
            ft += dt * grid_v[offset_x, offset_y].outer_product(derivative_weight[i, j])
        # ft = Round(ft, 3)
        F[i] = ft @ F[i]
        # 体積率推定
        J[i] = ti.math.determinant(F[i])

        J_copy[i] = J[i]
        F_copy[i] = F[i]


# Grid to Particle
@ti.kernel
def G2P():
    for i in range(n_particles):
        v[i] = [0.0, 0.0]
        for j in range(n_offset):
            offset_x = offset[i, j][0]
            offset_y = offset[i, j][1]
            # 速度情報を格子点から粒子へ
            v[i] += grid_v[offset_x, offset_y] * weight[i, j]


# 外力による速度計算
@ti.kernel
def velocity():
    # gravity[0] = gravity[0] + attractor_strength[None]
    for i in range(n_particles):
        # 重力の更新
        # g = gravity[0] + attractor_strength[None]
        # 重力
        v[i] += gravity[0] * dt


# 衝突計算
@ti.kernel
def conflict():
    for i in range(n_particles):
        # X
        if x[i][0] >= 0.999 or x[i][0] <= 0.001:
            v[i][0] = 0.0
        # Y
        if x[i][1] >= 0.999 or x[i][1] <= 0.001:
            v[i][1] = 0.0


# 座標計算
@ti.kernel
def update_location():
    for i in range(n_particles):
        x[i] += v[i] * dt


print("start")


def MPM(frame):
    decision_offset()
    # 1フレームの時だけ
    if frame == 1:
        estimate_Volume()  # 体積推定
    P2G()
    compute_stress()  # 応力テンソル計算
    update_velocity()  # 内力による速度更新
    update_Deformation_gradient()  # 変形勾配の更新
    G2P()
    velocity()
    conflict()
    update_location()


def main():
    frame, timestep = 0, 0
    wid_frame = gui.label("Frame")
    wid_frame.value = frame
    wid_time = gui.label("timestep")
    wid_time.value = timestep

    while gui.running:
        # Handle keyboard input
        for e in gui.get_events(gui.PRESS):
            if e.key == "f":
                gui.running = False
            elif e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            elif e.key == "w":
                gravity[0] = [0, 9]
            elif e.key == "s":
                gravity[0] = [0, -9]
            elif e.key == "d":
                gravity[0] = [9, 0]
            elif e.key == "a":
                gravity[0] = [-9, 0]
            elif e.key == "z":
                gravity[0] = [0, 0]

        # マウス位置取得
        mouse_pos = gui.get_cursor_pos()
        attractor_pos[None] = mouse_pos

        # 時間計算
        frame += 1
        timestep += 1
        wid_frame.value = frame
        wid_time.value = timestep

        # print(frame)
        MPM(frame)

        print("フレーム", frame)
        # print(grid_v[15, 15])
        print()

        # 格子点の描画
        gui.circles(grid_location_copy.to_numpy(), radius=6.0, color=0xb22222)
        # 格子線の描画
        gui.lines(begin=x_t, end=y_t, radius=1.0, color=0x068587)
        # 質点の描画
        gui.circles(x.to_numpy(), radius=2.0, color=0x808000)
        # マウスカーソル表示
        gui.circle(mouse_pos, radius=10, color=0x116699)
        # GUIの表示
        gui.show()


if __name__ == '__main__':
    main()
