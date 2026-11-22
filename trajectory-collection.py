import numpy as np
import json

from pathlib import Path

output_path = Path(__file__).resolve().parent / "teacher_trajectory.json"

def objective_function(x):
    return np.sum(x**2, axis=1)

class SimplePSO:
    def __init__(self, dim, pop_size, max_iter, interval_i):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.interval_i = interval_i
        self.trajectory = []
        self.best_fitness_history = []

    def optimize(self):
        pos = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        pbest = pos.copy()
        pbest_fitness = objective_function(pos)
        gbest = pos[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        for t in range(self.max_iter):
            r1, r2 = np.random.rand(), np.random.rand()
            vel = vel + r1 * (pbest - pos) + r2 * (gbest - pos)
            pos = pos + vel
            
            fitness = objective_function(pos)
            
            update_mask = fitness < pbest_fitness
            pbest[update_mask] = pos[update_mask]
            pbest_fitness[update_mask] = fitness[update_mask]
            
            if np.min(fitness) < gbest_fitness:
                gbest = pos[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            

            if (t + 1) % self.interval_i == 0:
                current_mean_fitness = np.mean(fitness)
                if len(self.best_fitness_history) == 0 or current_mean_fitness < self.best_fitness_history[-1]:
                    
                    pop_mean = np.mean(pos, axis=0)
                    pop_cov = np.cov(pos.T) 
                    
                    self.trajectory.append({
                        "generation": t + 1,
                        "mean": pop_mean.tolist(),
                        "covariance": pop_cov.tolist(),
                        "fitness": float(np.min(fitness))
                    })
                    self.best_fitness_history.append(current_mean_fitness)

        return self.trajectory

if __name__ == "__main__":
    pso = SimplePSO(dim=10, pop_size=50, max_iter=200, interval_i=5)
    
    print("开始运行优化器并收集轨迹...")
    teacher_trajectory = pso.optimize()
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(teacher_trajectory, f, indent=4)
    
    print(f"收集完毕！共提取了 {len(teacher_trajectory)} 个高质量轨迹节点。")
    print("数据已保存至 teacher_trajectory.json")