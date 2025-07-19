import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from flow_matching import NNvelocities, train_flow_matching, prepare_training_data, normalize_data, flow_integration
from prepare_data import prepare_flow_data, prepare_test_data

def simple_pca(data, n_components=2):
    # center the data
    data_centered = data - np.mean(data, axis=0)
    # compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    # eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # project data
    return data_centered @ eigenvectors[:, :n_components]

def plot_trajectories_pca(predicted_ends, actual_ends, test_starts, team_names):
    # combine all data for pca fitting
    all_data = np.vstack([test_starts, predicted_ends, actual_ends])
    
    # apply pca to all data
    pca_data = simple_pca(all_data[:, 2:], n_components=2)  # exclude team_id and season
    
    # split back into components
    n_teams = len(test_starts)
    starts_pca = pca_data[:n_teams]
    predicted_pca = pca_data[n_teams:2*n_teams]
    actual_pca = pca_data[2*n_teams:]
    
    plt.figure(figsize=(12, 8))
    
    # plot trajectories
    for i in range(n_teams):
        # actual trajectory (green)
        plt.plot([starts_pca[i, 0], actual_pca[i, 0]], 
                [starts_pca[i, 1], actual_pca[i, 1]], 
                'g-', alpha=0.6, linewidth=1)
        
        # predicted trajectory (red)
        plt.plot([starts_pca[i, 0], predicted_pca[i, 0]], 
                [starts_pca[i, 1], predicted_pca[i, 1]], 
                'r--', alpha=0.6, linewidth=1)
    
    # plot points
    plt.scatter(starts_pca[:, 0], starts_pca[:, 1], c='blue', s=50, alpha=0.7, label='early season')
    plt.scatter(actual_pca[:, 0], actual_pca[:, 1], c='green', s=50, alpha=0.7, label='actual late season')
    plt.scatter(predicted_pca[:, 0], predicted_pca[:, 1], c='red', s=50, alpha=0.7, marker='x', label='predicted late season')
    
    # add team labels for a few teams
    for i in range(0, n_teams, 4):  # every 4th team to avoid clutter
        plt.annotate(team_names[i], (starts_pca[i, 0], starts_pca[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.xlabel('pca component 1')
    plt.ylabel('pca component 2')
    plt.title('nfl team trajectories: predicted vs actual (pca space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nfl_trajectories_pca.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_predictions(predicted_ends, actual_ends):
    # calculate mean absolute error for each feature
    mae = np.mean(np.abs(predicted_ends - actual_ends), axis=0)
    overall_mae = np.mean(mae[2:])  # exclude team_id and season (constant features)
    return mae, overall_mae

def main():
    print("preparing training data...")
    p_train, q_train, team_mapping = prepare_flow_data()
    
    print("preparing flow matching training data...")
    T = 17
    x, t, v = prepare_training_data(p_train, q_train, T)
    x_norm, v_norm, x_mean, x_std, v_mean, v_std = normalize_data(x, v)
    
    print("training flow matching model...")
    model = NNvelocities(N=7, hidden=256)
    train_flow_matching(model, x_norm, t, v_norm, lr=1e-3, epochs=20000)
    
    print("preparing test data...")
    test_starts, test_ends, team_mapping = prepare_test_data()
    team_names = [team for team, _ in sorted(team_mapping.items(), key=lambda x: x[1])]
    
    print("making predictions...")
    sampled_times = [0.0, 1.0]  # start to end
    trajectories = flow_integration(model, test_starts, sampled_times, x_mean, x_std, v_mean, v_std)
    predicted_ends = trajectories[-1]  # final positions
    
    print("evaluating predictions...")
    mae, overall_mae = evaluate_predictions(predicted_ends, test_ends)
    
    print(f"\nresults:")
    print(f"overall mae: {overall_mae:.4f}")
    print(f"feature-wise mae:")
    features = ['team_id', 'season', 'epa_per_play', 'success_rate', 'pass_rate', 'air_yards', 'yards_per_play']
    for i, (feature, error) in enumerate(zip(features, mae)):
        print(f"  {feature}: {error:.4f}")
    
    print(f"\ntraining trajectories: {p_train.shape[0]}")
    print(f"test trajectories: {test_starts.shape[0]}")
    
    print("creating pca visualization...")
    plot_trajectories_pca(predicted_ends, test_ends, test_starts, team_names)

if __name__ == "__main__":
    main()