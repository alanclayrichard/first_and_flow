import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from flow_matching import NNvelocities, train_flow_matching, prepare_training_data, normalize_data
from prepare_data import prepare_flow_data

def predict_week(model, history, week, x_mean, x_std, v_mean, v_std, season_length):
    """
    predict the features at week t given history up to week t-1 using flow matching
    history: np.array of shape (t, N) containing features for weeks 0..t-1
    week: int, next week index (0-based) to predict
    season_length: total number of weeks in season (e.g., 17)
    returns: np.array of shape (N,) predicted features for week t
    """
    import torch
    # use last available week as starting point
    x_prev = torch.tensor(history[-1:], dtype=torch.float32)
    # normalized time for current position in season
    t_current = (len(history) - 1) / season_length
    # normalized time for next week
    t_next = week / season_length
    
    model.eval()
    with torch.no_grad():
        # integrate from current time to next time
        steps = max(1, int((t_next - t_current) * season_length * 10))  # more integration steps
        dt = (t_next - t_current) / steps
        
        x_t = x_prev
        t_curr = t_current
        
        for _ in range(steps):
            t_tensor = torch.full((x_t.shape[0], 1), t_curr, dtype=torch.float32)
            x_t_norm = (x_t - x_mean) / x_std
            v_norm = model(x_t_norm, t_tensor)
            v_pred = v_norm * v_std + v_mean
            x_t = x_t + v_pred * dt
            t_curr += dt
    
    return x_t.numpy().reshape(-1)

# train flow matching model on 2010-2023 data (now with full dataset)
p_train, q_train, team_map = prepare_flow_data()
print(f"training with {p_train.shape[0]} trajectories from {len(set([p[1] for p in p_train]))} seasons")
season_length = 17
x, t, v = prepare_training_data(p_train, q_train, season_length)
x_norm, v_norm, x_mean, x_std, v_mean, v_std = normalize_data(x, v)
model = NNvelocities(N=7, hidden=256)
train_flow_matching(model, x_norm, t, v_norm, lr=1e-4, epochs=5000)  # lower lr, fewer epochs

# load 2024 test data
df = pd.read_csv('nfl_testing_data.csv')
teams = sorted(df['team'].unique())

# prepare error tracking
features = ['team_id', 'season', 'epa_per_play', 'success_rate', 'pass_rate', 'air_yards', 'yards_per_play']
errors = {f: [] for f in features[2:]}
weeks = range(2, 19)  # include week 18

# iterate over weeks
for w in weeks:
    week_preds = []
    week_acts = []
    for team in teams:
        td = df[df['team']==team].sort_values('week')
        # check if we have data for week w
        if td[td['week']==w].empty:
            continue
        team_id = team_map[team] / 32
        season_norm = (2024 - 2010) / 13  # normalize by actual training range (2010-2023)
        # build history up to week w-1 (only use available weeks)
        hist = []
        for wk in range(1, w):
            week_data = td[td['week']==wk]
            if not week_data.empty:
                row = week_data.iloc[0]
                hist.append([
                    team_id, season_norm,
                    row.off_epa_per_play,
                    row.success_rate,
                    row.pass_rate,
                    row.avg_air_yards/20,
                    row.yards_per_play/10
                ])
        # need at least one week of history
        if len(hist) == 0:
            continue
        hist = np.array(hist)
        # use the correct week normalization for prediction
        pred = predict_week(model, hist, w-1, x_mean, x_std, v_mean, v_std, season_length)  # w-1 for 0-based indexing
        row = td[td['week']==w].iloc[0]
        act = np.array([
            team_id,
            season_norm,
            row.off_epa_per_play,
            row.success_rate,
            row.pass_rate,
            row.avg_air_yards/20,
            row.yards_per_play/10
        ])
        week_preds.append(pred)
        week_acts.append(act)
    # compute mean absolute error across teams
    if not week_preds:
        # no predictions this week
        for f in features[2:]:
            errors[f].append(np.nan)
        continue
    # stack predictions and actuals into 2D arrays (n_teams, n_features)
    wp = np.vstack(week_preds)
    wa = np.vstack(week_acts)
    if wp.shape != wa.shape:
        # mismatched shapes
        for f in features[2:]:
            errors[f].append(np.nan)
        continue
    mae = np.mean(np.abs(wp - wa), axis=0)
    # append per-feature MAE (skip team_id & season)
    for j, f in enumerate(features[2:], start=2):
        errors[f].append(mae[j])

# plot MAE by week
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
for f, vals in errors.items():
    plt.plot(weeks, vals, marker='o', label=f)
plt.title('weekly prediction MAE (2024)')
plt.xlabel('week')
plt.ylabel('mean absolute error')
plt.xticks(weeks)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('weekly_mae.png', dpi=150)
plt.show()
