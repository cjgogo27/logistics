import numpy as np
import pandas as pd
import random
from typing import List, Tuple

H1 = 1.2
H2 = 2.4
H3 = 1.83
N = 52
M = 2600
n_k = 50

a_coords = np.arrange(14)
x_coords = np.arange(21)
y_coords = np.arange(51)
z_coords = np.arange(5)

x_grid, y_grid, z_grid,a_grid = np.meshgrid(a_coords,x_coords, y_coords, z_coords, indexing='ij')
a_coords_flat = a_grid.flatten()
x_coords_flat = x_grid.flatten()
y_coords_flat = y_grid.flatten()
z_coords_flat = z_grid.flatten()


data_A = pd.read_excel('D:\物流设计\附件A.xlsx')
p_k_i = data_A.iloc[:, 2].to_numpy()
q_k_i = data_A.iloc[:, 3].to_numpy()

s_k_i_l_j_df = pd.read_excel('D:\物流设计\布料关联度矩阵-附件B.xlsx')
s_k_i_l_j = s_k_i_l_j_df .to_numpy()

def calculate_de():
    de = np.zeros(M)
    for e in range(M):
        if z_coords_flat[e] < 2:
            de[e] = a_coords_flat[e] * H1 + (np.abs(x_coords_flat[e] - 9) + np.abs(x_coords_flat[e] - 5)) * H2 + y_coords_flat[e] * H3
        else:
            de[e] = a_coords_flat[e] * H1 + (np.abs(x_coords_flat[e] - 9) + np.abs(x_coords_flat[e] - 5)) * H2 + y_coords_flat[
                e] * H3 + 75.9
    return de


de = calculate_de()


class Particle:
    def __init__(self, position=None):
        if position is None:
            self.position = np.random.permutation(M)[:N * n_k].reshape(N, n_k)
        else:
            self.position = position

        self.velocity = np.random.randint(-5, 6, size=(N, n_k))

        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('-inf')

        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        e_idx = self.position.flatten()
        f1 = np.sum((0.5 * p_k_i + 0.5 * q_k_i) * de[e_idx])

        threshold = 0.8
        idx = np.where(s_k_i_l_j > threshold)
        sample_size = 1000
        if len(idx[0]) > sample_size:
            sampled = np.random.choice(len(idx[0]), sample_size, replace=False)
            idx = tuple(i[sampled] for i in idx)
        if len(idx[0]) > 0:
            e_pos = self.position[idx[0], idx[1]]
            f_pos = self.position[idx[2], idx[3]]
            dx = x_coords_flat[e_pos] - x_coords_flat[f_pos]
            dy = y_coords_flat[e_pos] - y_coords_flat[f_pos]
            dz = z_coords_flat[e_pos] - z_coords_flat[f_pos]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            f2 = np.sum(s_k_i_l_j[idx] * (1 / (1 + dist)))
        else:
            f2 = 0

        x_center = np.mean(x_coords_flat[self.position], axis=1)
        y_center = np.mean(y_coords_flat[self.position], axis=1)
        z_center = np.mean(z_coords_flat[self.position], axis=1)
        dx = x_coords_flat[self.position] - x_center[:, None]
        dy = y_coords_flat[self.position] - y_center[:, None]
        dz = z_coords_flat[self.position] - z_center[:, None]
        f3 = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

        w1 = 0.4
        w2 = 0.3
        w3 = 0.3

        fitness = w2 * f2 - w1 * f1 - w3 * f3
        return fitness

    def update_velocity(self, gbest_position, w=0.7, c1=1.4, c2=1.4):
        r1, r2 = random.random(), random.random()

        cognitive = c1 * r1 * (self.pbest_position - self.position)

        social = c2 * r2 * (gbest_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        self.velocity = np.clip(self.velocity, -10, 10)

    def update_position(self):
        new_position = self.position + self.velocity.astype(int)

        for k in range(N):
            for i in range(n_k):
                new_position[k, i] = max(0, min(new_position[k, i], M - 1))

                if len(np.unique(new_position)) < N * n_k:
                    used = set(new_position.flatten())
                    available = [e for e in range(M) if e not in used]
                    if available:
                        new_position[k, i] = random.choice(available)

        self.position = new_position

        self.fitness = self.calculate_fitness()

        if self.fitness > self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()


def particle_swarm_optimization(num_particles=30, max_iter=100, w=0.7, c1=1.4, c2=1.4):
    swarm = [Particle() for _ in range(num_particles)]

    gbest_particle = max(swarm, key=lambda p: p.fitness)
    gbest_position = gbest_particle.position.copy()
    gbest_fitness = gbest_particle.fitness

    for iteration in range(max_iter):
        for particle in swarm:
            particle.update_velocity(gbest_position, w, c1, c2)
            particle.update_position()

            if particle.fitness > gbest_fitness:
                gbest_fitness = particle.fitness
                gbest_position = particle.position.copy()

        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration + 1}/{max_iter}")

    return gbest_position, gbest_fitness


def convert_to_decision_matrix(position):
    x_kie = np.zeros((N, n_k, M), dtype=int)
    for k in range(N):
        for i in range(n_k):
            e = position[k, i]
            x_kie[k, i, e] = 1
    return x_kie


print("开始使用粒子群算法求解...")
best_position, best_fitness = particle_swarm_optimization(num_particles=5, max_iter=10)

x_kie = convert_to_decision_matrix(best_position)


def check_constraints(x_kie):

    for e in range(M):
        if np.sum(x_kie[:, :, e]) > 1:
            return False

    for k in range(N):
        for i in range(n_k):
            if np.sum(x_kie[k, i, :]) != 1:
                return False

    return True

print("\n优化完成!")
print(f"约束条件满足: {check_constraints(x_kie)}")

np.save('warehouse_allocation.npy', x_kie)
print("\n决策矩阵已保存到 'warehouse_allocation.npy'")

print("\n部分分配结果示例:")
for k in range(3):
    for i in range(2):
        e_allocated = np.where(x_kie[k, i, :] == 1)[0][0]
        print(f"面料 {k}, 布料 {i} --> 货位 "
              f"({x_coords_flat[e_allocated]}, {y_coords_flat[e_allocated]}, {z_coords_flat[e_allocated]})")