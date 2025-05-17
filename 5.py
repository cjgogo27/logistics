import numpy as np
import pandas as pd
import random
from typing import List, Tuple

# 模型参数
H1 = 1.2
H2 = 2.4
H3 = 1.83
N = 52  # 面料类别数
M = 2600  # 货位总数
n_k = 50  # 每类面料中颜色布料数量（假设为常数）

# 假设x_coords, y_coords, z_coords是表示货位坐标的一维数组
x_coords = np.arange(21)  # 假设从0到20，可根据实际情况调整
y_coords = np.arange(51)  # 假设从0到50，可根据实际情况调整
z_coords = np.arange(5)  # 假设从0到4，可根据实际情况调整

# 生成网格坐标
x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
x_coords_flat = x_grid.flatten()
y_coords_flat = y_grid.flatten()
z_coords_flat = z_grid.flatten()

# 示例：随机生成p_k_i和q_k_i数据（实际应用中需替换为真实数据）
np.random.seed(42)
p_k_i = np.random.rand(N * n_k)
q_k_i = np.random.rand(N * n_k)

# 示例：随机生成关联度矩阵s_k_i_l_j（实际应用中需替换为真实数据）
s_k_i_l_j = np.random.rand(N, n_k, N, n_k)


# 计算每个货位的d_e值
def calculate_de():
    de = np.zeros(M)
    for e in range(M):
        if z_coords_flat[e] < 2:
            de[e] = 10 * H1 + (np.abs(x_coords_flat[e] - 9) + np.abs(x_coords_flat[e] - 5)) * H2 + y_coords_flat[e] * H3
        else:
            de[e] = 10 * H1 + (np.abs(x_coords_flat[e] - 9) + np.abs(x_coords_flat[e] - 5)) * H2 + y_coords_flat[
                e] * H3 + 75.9
    return de


de = calculate_de()


# 粒子群算法实现
class Particle:
    def __init__(self, position=None):
        # 粒子位置 - 每个布料分配的货位
        if position is None:
            # 随机初始化，确保每个布料分配一个唯一货位
            self.position = np.random.permutation(M)[:N * n_k].reshape(N, n_k)
        else:
            self.position = position

        # 粒子速度
        self.velocity = np.random.randint(-5, 6, size=(N, n_k))

        # 个体最优位置和适应度
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('-inf')

        # 当前适应度
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        """计算粒子的适应度值"""
        # 目标函数1：最小化总搬运距离
        f1 = 0
        for k in range(N):
            for i in range(n_k):
                e = self.position[k, i]
                f1 += (0.5 * p_k_i[k * n_k + i] + 0.5 * q_k_i[k * n_k + i]) * de[e]

        # 目标函数2：最大化关联度高的布料之间的邻近度
        f2 = 0
        for k in range(N):
            for i in range(n_k):
                for l in range(N):
                    for j in range(n_k):
                        if k == l and i == j:
                            continue
                        e = self.position[k, i]
                        f = self.position[l, j]
                        distance = np.sqrt((x_coords_flat[e] - x_coords_flat[f]) ** 2 +
                                           (y_coords_flat[e] - y_coords_flat[f]) ** 2 +
                                           (z_coords_flat[e] - z_coords_flat[f]) ** 2)
                        f2 += s_k_i_l_j[k, i, l, j] * (1 / (1 + distance))  # 距离越近，值越大

        # 目标函数3：最小化同类布料之间的距离
        f3 = 0
        for k in range(N):
            # 计算类别k的中心坐标
            x_center = np.mean([x_coords_flat[self.position[k, i]] for i in range(n_k)])
            y_center = np.mean([y_coords_flat[self.position[k, i]] for i in range(n_k)])
            z_center = np.mean([z_coords_flat[self.position[k, i]] for i in range(n_k)])

            # 计算类内距离
            for i in range(n_k):
                e = self.position[k, i]
                dist = np.sqrt((x_coords_flat[e] - x_center) ** 2 +
                               (y_coords_flat[e] - y_center) ** 2 +
                               (z_coords_flat[e] - z_center) ** 2)
                f3 += dist

        # 设置多目标权重
        w1 = 0.4  # 搬运效率权重
        w2 = 0.3  # 关联度权重
        w3 = 0.3  # 类内聚集度权重

        # 综合适应度（最大化）
        fitness = w2 * f2 - w1 * f1 - w3 * f3
        return fitness

    def update_velocity(self, gbest_position, w=0.7, c1=1.4, c2=1.4):
        """更新粒子速度"""
        r1, r2 = random.random(), random.random()

        # 认知部分 - 向个体最优学习
        cognitive = c1 * r1 * (self.pbest_position - self.position)

        # 社会部分 - 向全局最优学习
        social = c2 * r2 * (gbest_position - self.position)

        # 更新速度
        self.velocity = w * self.velocity + cognitive + social

        # 限制速度范围
        self.velocity = np.clip(self.velocity, -10, 10)

    def update_position(self):
        """更新粒子位置"""
        # 应用速度
        new_position = self.position + self.velocity.astype(int)

        # 确保位置在有效范围内
        for k in range(N):
            for i in range(n_k):
                # 确保货位索引在有效范围内
                new_position[k, i] = max(0, min(new_position[k, i], M - 1))

                # 处理重复分配问题（简化版：如果有重复，随机找一个未使用的货位）
                if len(np.unique(new_position)) < N * n_k:
                    used = set(new_position.flatten())
                    available = [e for e in range(M) if e not in used]
                    if available:
                        new_position[k, i] = random.choice(available)

        self.position = new_position

        # 重新计算适应度
        self.fitness = self.calculate_fitness()

        # 更新个体最优
        if self.fitness > self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()


def particle_swarm_optimization(num_particles=30, max_iter=100, w=0.7, c1=1.4, c2=1.4):
    """粒子群优化算法"""
    # 初始化粒子群
    swarm = [Particle() for _ in range(num_particles)]

    # 初始化全局最优
    gbest_particle = max(swarm, key=lambda p: p.fitness)
    gbest_position = gbest_particle.position.copy()
    gbest_fitness = gbest_particle.fitness

    # 迭代优化
    for iteration in range(max_iter):
        for particle in swarm:
            # 更新速度和位置
            particle.update_velocity(gbest_position, w, c1, c2)
            particle.update_position()

            # 更新全局最优
            if particle.fitness > gbest_fitness:
                gbest_fitness = particle.fitness
                gbest_position = particle.position.copy()

        # 打印进度
        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration + 1}/{max_iter}, 全局最优适应度: {gbest_fitness}")

    return gbest_position, gbest_fitness


# 将货位分配结果转换为0-1决策矩阵
def convert_to_decision_matrix(position):
    """将货位分配结果转换为0-1决策矩阵 x_kie"""
    x_kie = np.zeros((N, n_k, M), dtype=int)
    for k in range(N):
        for i in range(n_k):
            e = position[k, i]
            x_kie[k, i, e] = 1
    return x_kie


# 运行PSO算法
print("开始使用粒子群算法求解...")
best_position, best_fitness = particle_swarm_optimization(num_particles=20, max_iter=50)

# 转换为0-1决策矩阵
x_kie = convert_to_decision_matrix(best_position)


# 验证约束条件
def check_constraints(x_kie):
    """检查约束条件是否满足"""
    # 约束1：每个货位最多存放一种布料
    for e in range(M):
        if np.sum(x_kie[:, :, e]) > 1:
            return False

    # 约束2：每种布料必须存放在一个货位上
    for k in range(N):
        for i in range(n_k):
            if np.sum(x_kie[k, i, :]) != 1:
                return False

    return True


# 输出结果
print("\n优化完成!")
print(f"最优适应度值: {best_fitness}")
print(f"约束条件满足: {check_constraints(x_kie)}")

# 保存结果
np.save('warehouse_allocation.npy', x_kie)
print("\n决策矩阵已保存到 'warehouse_allocation.npy'")

# 输出一些分配结果示例
print("\n部分分配结果示例:")
for k in range(3):  # 只显示前3类
    for i in range(2):  # 每类只显示前2个
        e_allocated = np.where(x_kie[k, i, :] == 1)[0][0]
        print(f"面料类别 {k}, 颜色 {i} --> 货位 {e_allocated} "
              f"({x_coords_flat[e_allocated]}, {y_coords_flat[e_allocated]}, {z_coords_flat[e_allocated]})")