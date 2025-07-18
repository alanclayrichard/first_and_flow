import numpy as np
import torch
from matplotlib import pyplot as plt

# generate distributions
def generate_distributions(M=1000, N=2):
    p = np.random.normal(size=(M, N))
    q = np.zeros(shape=(M, N))
    q[:, 0] = np.linspace(9, 10, M)
    q[:, 1] = np.linspace(10, 9, M)
    p = p * (2 - 1) + 1
    return p, q

# get the positions 
def get_positions(p, q, T):
    M, N = p.shape
    positions = np.zeros((M, N, T + 1))
    for t in range(T + 1):
        dt = t / T  
        positions[:, :, t] = (1 - dt) * p + dt * q
    return positions

# velocity field
def get_velocities(positions, T):
    M, N = positions[:, :, 0].shape
    velocities = np.zeros((M, N, T))
    dt = 1 / T
    for t in range(T):
        velocities[:, :, t] = (positions[:, :, t + 1] - positions[:, :, t]) / dt
    return velocities

class NNvelocities(torch.nn.Module):
    def __init__(self, N, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, N)
        )

    def forward(self, x, t):
        input = torch.cat([x, t], dim=-1)
        return self.net(input)

def train_flow_matching(model, x, t, v, lr=1e-3, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        v_pred = model(x, t)
        loss = loss_function(v_pred, v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss= {loss.item()}")

def prepare_training_data(p, q, T):
    positions = get_positions(p, q, T)
    velocities = get_velocities(positions, T)
    t_np = np.linspace(0, 1, T)
    t_np = np.tile(t_np, (p.shape[0], 1))
    t_np = t_np[..., np.newaxis]

    positions = torch.tensor(positions[:, :, :-1], dtype=torch.float32)
    velocities = torch.tensor(velocities, dtype=torch.float32)
    time = torch.tensor(t_np, dtype=torch.float32)

    x = positions.permute(0, 2, 1).reshape(-1, p.shape[1])
    t = time.reshape(-1, 1)
    v = velocities.permute(0, 2, 1).reshape(-1, p.shape[1])

    return x, t, v

def normalize_data(x, v):
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)
    x_norm = (x - x_mean) / x_std

    v_mean = v.mean(dim=0, keepdim=True)
    v_std = v.std(dim=0, keepdim=True)
    v_norm = (v - v_mean) / v_std

    return x_norm, v_norm, x_mean, x_std, v_mean, v_std

def flow_integration(model, p, sampled_times, x_mean, x_std, v_mean, v_std, steps_per_unit=100):
    x_t = torch.tensor(p, dtype=torch.float32)
    trajectories = [x_t.numpy()]
    
    model.eval()
    with torch.no_grad():
        t_curr = 0.0
        for i in range(1, len(sampled_times)):
            t_next = sampled_times[i]
            steps = int((t_next - t_curr) * steps_per_unit)
            dt = (t_next - t_curr) / steps

            for _ in range(steps):
                t_tensor = torch.full((x_t.shape[0], 1), t_curr, dtype=torch.float32)
                x_t_norm = (x_t - x_mean) / x_std
                v_pred_norm = model(x_t_norm, t_tensor)
                v_pred = v_pred_norm * v_std + v_mean
                x_t = x_t + v_pred * dt
                t_curr += dt

            trajectories.append(x_t.numpy())

    return trajectories

def plot_trajectories(trajectories, sampled_times):
    colors = ['green', 'blue', 'orange', 'purple', 'red']
    labels = [f't={t:.2f}' for t in sampled_times]

    plt.figure(figsize=(8, 8))
    for traj, color, label in zip(trajectories, colors, labels):
        plt.scatter(traj[:, 0], traj[:, 1], s=3, color=color, label=label)

    plt.title("Flow Matching - Positions at Sampled Time Steps")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# main section
if __name__ == "__main__":
    # parameters
    N, M, T = 2, 1000, 100
    sampled_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    # generate toy data
    p, q = generate_distributions(M, N)
    x, t, v = prepare_training_data(p, q, T)
    x, v, x_mean, x_std, v_mean, v_std = normalize_data(x, v)

    # train model
    model = NNvelocities(N)
    train_flow_matching(model, x, t, v)

    # inference and plotting
    trajectories = flow_integration(model, p, sampled_times, x_mean, x_std, v_mean, v_std)
    plot_trajectories(trajectories, sampled_times)
