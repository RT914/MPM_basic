import numpy as np
import taichi as ti
from numpy import linalg
import random as ra
import matplotlib.cm as cm


ra.seed(313)

ti.init(arch=ti.gpu)

# Window Size
n = 500
gui = ti.GUI("MPM", res=(n, n), background_color=0xffffff, fullscreen=False)

maxSimulationTime = 10000

n_particles: int = 3000
n_grid = 30
dimensions = 2

difference = 1 / n_grid
dt = 1.0e-4
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
    2, dtype=float, shape=n_grid*n_grid
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
    2, dtype=float, shape=n_grid*n_grid
)
vector_lines = ti.Vector.field(
    2, dtype=float, shape=n_grid*n_grid
)
grid_v_copy = ti.Vector.field(
    2, dtype=float, shape=n_grid*n_grid
)
grid_mass_copy = ti.field(
    dtype=float, shape=n_grid*n_grid
)
offset = ti.Vector.field(
    2, dtype=float, shape=(n_particles, 4)
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


# カラーコード作成
def make_colorcode_for_J(j):
    # c_array = np.array([])
    c_array = np.zeros(n_particles, dtype=np.int32)
    for c in range(n_particles):
        ce = np.log(j[c]) / np.log(2.0) + 0.5
        string = ""
        result = "0x"

        red = int(round(cm.bwr(ce)[0] * 255))
        green = int(round(cm.bwr(ce)[1] * 255))
        blue = int(round(cm.bwr(ce)[2] * 255))

        color = blue << 16 | green << 8 | red
        c_array[c] = color
        # = np.append(c_array, color)

    """
        for d in range(3):
            if round(cm.bwr(ce)[d] * 255) < 16:
                string += "0"
            string += hex(round(cm.bwr(ce)[d] * 255))[2:]
            # print(string)
        result += string
        c_array = np.append(c_array, hex(int(result, 16)))
        """

    return c_array


# カラーコード作成
def make_colorcode_for_grid_mass(gm):
    c_array = np.zeros(n_grid*n_grid, dtype=np.int32)
    for c in range(n_grid*n_grid):
        ce = gm[c]

        red = int(round(cm.Greys(ce)[0] * 255))
        green = int(round(cm.Greys(ce)[1] * 255))
        blue = int(round(cm.Greys(ce)[2] * 255))

        color = blue << 16 | green << 8 | red
        c_array[c] = color
    return c_array


# 格子線
begin = []
end = []
# 縦
for a in range(n_grid):
    begin.append(1.0/n_grid * a)
    begin.append(0.0)
    end.append(1.0/n_grid * a)
    end.append(1.0)

# 横
for a in range(n_grid):
    begin.append(0.0)
    begin.append(1.0/n_grid * a)
    end.append(1.0)
    end.append(1.0/n_grid * a)

x_t = np.reshape(begin, (n_grid*2, 2))
y_t = np.reshape(end, (n_grid*2, 2))

# 格子点の位置設定
grid_num = 0
for a in range(n_grid):
    for b in range(n_grid):
        grid_location[a, b] = [a / n_grid, b / n_grid]
        grid_location_copy[grid_num] = grid_location[a, b]
        grid_num += 1


# 四捨五入
@ti.func
def Round(matrix, num):
    m = ti.round(matrix * pow(10, (num-1.0)))
    return m / pow(10, (num-1.0))


# grid statusの初期化
@ti.func
def initialization_grid(num1, num2):
    grid_v[num1, num2] = [0.0, 0.0]
    grid_mass[num1, num2] = 0.0
    grid_momentum[num1, num2] = [0.0, 0.0]
    f[num1, num2] = [0.0, 0.0]


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


"""
# 重みづけ関数その1
@ti.func
def Weighting(dist: float) -> float:
    dist = ti.abs(dist)
    kernel_value = 0.0
    if 0.0 <= dist < 1.0:
        kernel_value = 1.0 - dist
    else:
        kernel_value = 0.0
    return kernel_value


# 重みづけ関数その2
@ti.func
def Weighting_dash(dist: float) -> float:
    sgn = 1.0 if dist >= 0.0 else -1.0
    dist = ti.abs(dist)
    kernel_value = 0.0
    if 0.0 <= dist < 1.0:
        kernel_value = -1.0
    else:
        kernel_value = 0.0
    return kernel_value * sgn


@ti.func
def Weighting(dist: float) -> float:
    dist = ti.abs(dist)
    kernel_value = 0.0
    if 0.0 <= dist < 0.5:
        kernel_value = 0.75 - dist**2
    elif 0.5 <= dist < 1.5:
        kernel_value = 0.5 * (1.5 - dist)**2
    else:
        kernel_value = 0.0
    return kernel_value


@ti.func
def Weighting_dash(dist: float) -> float:
    sgn = 1.0 if dist >= 0.0 else -1.0
    dist = ti.abs(dist)
    kernel_value = 0.0
    if 0.0 <= dist < 0.5:
        kernel_value = -2 * dist
    elif 0.5 <= dist < 1.5:
        kernel_value = - 1.5 + dist
    else:
        kernel_value = 0.0
    return kernel_value * sgn
"""


@ti.func
def print_weight(w):
    print(w)


# Particle to Grid
@ti.kernel
def P2G():
    for i in range(n_grid):
        for j in range(n_grid):

            # grid statusの初期化
            initialization_grid(i, j)

            for k in range(n_particles):

                # 重み
                weight = \
                    Weighting((x[k][0] - grid_location[i, j][0]) / difference) * \
                    Weighting((x[k][1] - grid_location[i, j][1]) / difference)
                # print("weight", weight)
                # 運動量情報を粒子から格子点へ
                grid_momentum[i, j] += mass[k] * v[k] * weight
                # 質量情報を粒子から格子点へ
                grid_mass[i, j] += mass[k] * weight
                # print(grid_momentum)
                # print(grid_mass)

            if grid_mass[i, j] > 0:
                grid_v[i, j] = grid_momentum[i, j] / grid_mass[i, j]
                grid_mass_copy[n_grid * i + j] = grid_mass[i, j]


# 体積推定
@ti.kernel
def estimate_Volume():
    for i in range(n_particles):
        num = 0.0
        weight_value = 0.0
        for j in range(n_grid):
            for k in range(n_grid):
                # 重み
                weight = \
                    Weighting((x[i][0] - grid_location[j, k][0]) / difference) * \
                    Weighting((x[i][1] - grid_location[j, k][1]) / difference)

                weight_value += weight
                # 体積推定
                num += grid_mass[j, k] * weight * difference**2
        p_volume[i] = mass[i] / num
        print_weight(weight_value)


# 応力テンソル計算
@ti.kernel
def compute_stress():
    for i in range(n_particles):
        #  Piola-Kirchhoff
        piola = micro * (F[i] - F[i].transpose().inverse()) + lamda * ti.log(J[i]) * F[i].transpose().inverse()
        # print("J", J[i])
        # print("log", ti.log(J[i]))
        # print(piola * trans_F / J[i])
        # 応力
        sigma[i] = (1 / J[i]) * (piola @ F[i].transpose())
        # sigma[i] = micro * (F[i] @ trans_f)
        # sigma[i] = \
        #    micro * (F[i] @ trans_f - ti.Matrix.identity(float, 2)) / J[i]\
        #    + lamda * ti.log(J[i]) * ti.Matrix.identity(float, 2) / J[i]
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
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_particles):
                # 重み
                weight_x = \
                    Weighting_dash((x[k][0] - grid_location[i, j][0]) / difference) * \
                    Weighting((x[k][1] - grid_location[i, j][1]) / difference) / \
                    difference

                weight_y = \
                    Weighting((x[k][0] - grid_location[i, j][0]) / difference) * \
                    Weighting_dash((x[k][1] - grid_location[i, j][1]) / difference) / \
                    difference

                weight_vector[k] = [weight_x, weight_y]

                f[i, j][0] += \
                    p_volume[k] * J[k] * \
                    (sigma[k][0, 0] * weight_vector[k][0] + sigma[k][0, 1] * weight_vector[k][1]) * \
                    (-1)

                f[i, j][1] += \
                    p_volume[k] * J[k] * \
                    (sigma[k][1, 0] * weight_vector[k][0] + sigma[k][1, 1] * weight_vector[k][1]) * \
                    (-1)
            f_divide_m_0 = 0.0
            f_divide_m_1 = 0.0
            f_grid_copy[n_grid*i+j] = f[i, j]/100 + grid_location_copy[n_grid*i+j]
            # print("質量", grid_mass[i, j])
            # f/m
            if f[i, j][0] == 0.0 or grid_mass[i, j] <= 1.0e-5:
                f_divide_m_0 = 0.0
            else:
                f_divide_m_0 = f[i, j][0] / grid_mass[i, j]

            if f[i, j][1] == 0.0 or grid_mass[i, j] <= 1.0e-5:
                f_divide_m_1 = 0.0
            else:
                f_divide_m_1 = f[i, j][1] / grid_mass[i, j]

            f_divide_m[0] = [f_divide_m_0, f_divide_m_1]
            print(f_divide_m[0])
            # 格子点の速度更新
            grid_v[i, j] = grid_v[i, j] + dt * f_divide_m[0]

            # 格子点速度のコピー
            grid_v_copy[n_grid*i+j] = grid_v[i, j]
            vector_lines[n_grid*i+j] = grid_location_copy[n_grid*i+j] + grid_v_copy[n_grid*i+j]
            # print(grid_location_copy[10*i+j], i, j, 10*i+j)
            # print(grid_v[i, j])


# 変形勾配の更新
@ti.kernel
def update_Deformation_gradient():
    for i in range(n_particles):
        ft = ti.Matrix.identity(float, 2)
        for j in range(n_grid):
            for k in range(n_grid):
                # 重み
                weight_x = \
                    Weighting_dash((x[i][0] - grid_location[j, k][0]) / difference) * \
                    Weighting((x[i][1] - grid_location[j, k][1]) / difference) / \
                    difference

                weight_y = \
                    Weighting((x[i][0] - grid_location[j, k][0]) / difference) * \
                    Weighting_dash((x[i][1] - grid_location[j, k][1]) / difference) / \
                    difference

                weight_vector[i] = [weight_x, weight_y]
                # print("速度", grid_v[j, k])
                # print("座標", grid_location[j, k])
                # print(weight_vector[i])
                # print(grid_v[j, k].outer_product(weight_vector[i]))
                ft += dt * grid_v[j, k].outer_product(weight_vector[i])
                # if grid_v[j, k][1] < -0.001:
                    # print(grid_v[j, k])

        # ft = Round(ft, 3)
        F[i] = ft @ F[i]
        # 体積率推定
        J[i] = ti.math.determinant(F[i])
        # print("ft", ft)
        # F[i] = [[0.99, 0.01], [0.01, 0.99]]
        # print(F[i])
        J_copy[i] = J[i]
        F_copy[i] = F[i]


# Grid to Particle
@ti.kernel
def G2P():
    for i in range(n_particles):
        v[i] = [0.0, 0.0]
        for j in range(n_grid):
            for k in range(n_grid):
                # 重み
                weight = \
                    Weighting((x[i][0] - grid_location[j, k][0]) / difference) * \
                    Weighting((x[i][1] - grid_location[j, k][1]) / difference)
                # 速度情報を格子点から粒子へ
                v[i] += grid_v[j, k] * weight


# 外力による速度計算
@ti.kernel
def velocity():
    for i in range(n_particles):
        # 重力
        v[i] += gravity[0] * dt

        """
        # 速度制限
        # X
        if abs(v[i][0]) >= limit_v:
            if v[i][0] >= 0.0:
                v[i][0] = limit_v
            else:
                v[i][0] = (-1) * limit_v
        # Y
        if abs(v[i][1]) >= limit_v:
            if v[i][1] >= 0.0:
                v[i][1] = limit_v
            else:
                v[i][1] = (-1) * limit_v
        # print("速度：", v[i])
        """


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
    P2G()  # Particle to Grid

    # 1フレーム字の時だけ
    if frame == 1:
        estimate_Volume()  # 体積推定

    compute_stress()  # 応力テンソル計算
    update_velocity()  # 内力による速度更新
    update_Deformation_gradient()  # 変形勾配の更新

    """
    for e in range(n_grid):
        for e2 in range(n_grid):
            print("格子速度：", e, e2, grid_v[e, e2])
    """
    G2P()  # Grid to Particle
    velocity()  # 外力による速度計算
    conflict()  # 衝突計算
    """
    for e in range(n_particles):
        print("速度：", v[e])
        """
    update_location()  # 座標計算


def main():
    frame, timestep = 0, 0
    wid_frame = gui.label("Frame")
    wid_frame.value = frame
    wid_time = gui.label("timestep")
    wid_time.value = timestep

    for t in range(maxSimulationTime):
        # Handle keyboard input
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "f":
                break
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

        # 時間計算
        frame += 1
        timestep += 1
        wid_frame.value = frame
        wid_time.value = timestep

        # print(frame)
        MPM(frame)

        print("フレーム", frame)
        print()

        """
        for u in range(n_particles):
            print("粒子No.", u+1)
            print("ストレス", sigma[u])
            print("J", J_copy[u])
            print("速度", v[u])
            # print("F", F_copy[u])
            # print("座標", x[u])
            print()
        """

        # 格子点の描画
        # colorcode_for_gm = make_colorcode_for_grid_mass(grid_mass_copy)
        # gui.circles(grid_location_copy.to_numpy(), radius=6.0, color=colorcode_for_gm)
        # 格子線の描画
        gui.lines(begin=x_t, end=y_t, radius=1.0, color=0x068587)
        # 格子点の力の描画
        gui.lines(begin=grid_location_copy.to_numpy(), end=f_grid_copy.to_numpy(), radius=2.0, color=0x808000)
        # 粒子のストレスの描画
        # gui.lines(begin=x.to_numpy(), end=sigma_x.to_numpy(), radius=1.0, color=0xb22222)
        # gui.lines(begin=x.to_numpy(), end=sigma_y.to_numpy(), radius=1.0, color=0xff8c00)
        # 格子点速度の描画
        # gui.lines(begin=grid_location_copy.to_numpy(), end=vector_lines.to_numpy(), radius=2.0, color=0x808000)
        # 質点の描画
        # colorcode_for_J = make_colorcode_for_J(J_copy)
        # gui.circles(x.to_numpy(), radius=6.0, color=colorcode_for_J)

        gui.circles(x.to_numpy(), radius=2.0, color=0x808000)

        gui.show()


if __name__ == '__main__':
    main()
