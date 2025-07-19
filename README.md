# first_and_flow

flow matching model to predict nfl team performance trajectories over the course of a season

## setup
```bash
pip install pandas nflfastpy matplotlib torch numpy
python download_data.py
python train_model.py
```

## data
- training: 2010-2023 nfl team weekly performance (6 features per team per week)
- testing: 2024 season data
- features: off_epa_per_play, success_rate, pass_rate, avg_air_yards, yards_per_play, total_plays

## model
flow matching architecture that learns team performance state transitions week-to-week