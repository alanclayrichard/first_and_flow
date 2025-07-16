# first_and_flow
Flow matching for football

## Overview

**First and Flow** is a machine learning framework that applies flow matching techniques to model NFL team performance dynamics. The project combines robust data processing pipelines with cutting-edge generative modeling to understand and predict team performance trajectories.

## Project Structure

```
first_and_flow/
├── data/                           # Data generation and processing
│   ├── generate_data.R            # Main data pipeline
│   ├── nfl_data_processor.R       # Core NFL data processing
│   ├── test_processor.R           # Validation script
│   ├── install_requirements.R     # R package setup
│   ├── usage_examples.R           # Usage documentation
│   ├── raw/                       # Raw downloaded data (cached)
│   └── processed/                 # Processed training data
│       └── nfl_flow_training_data.csv
├── src/                           # Flow matching model implementation
│   ├── first_and_flow/           # Main Python package
│   │   ├── __init__.py           # Package initialization
│   │   ├── data_loader.py        # Data loading and preprocessing
│   │   ├── flow_model.py         # Flow matching model architecture
│   │   ├── trainer.py            # Training loop and optimization
│   │   └── evaluator.py          # Model evaluation and analysis
│   └── main.py                   # Main training/evaluation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Data Generation

```bash
# Install R packages
cd data
Rscript install_requirements.R

# Test data pipeline
Rscript test_processor.R

# Generate training data (2018-2023 by default)
Rscript generate_data.R

# Or specify custom years
Rscript generate_data.R 2020 2023
```

### 2. Model Training

```bash
# Install Python dependencies
pip install -r requirements.txt

# Train flow matching model
cd src
python main.py --epochs 100 --batch-size 32

# Train with custom parameters
python main.py --epochs 200 --hidden-dim 256 --learning-rate 5e-4
```

### 3. Model Evaluation

```bash
# Evaluate trained model
python main.py --eval-only

# Use specific model checkpoint
python main.py --eval-only --model-path checkpoints/best_model.pt
```

## Features

### 🏈 **NFL Data Processing**
- **Comprehensive data pipeline** using nflfastR package
- **Automated feature engineering** for team performance metrics
- **Temporal consistency** with proper train/validation/test splits
- **Quality validation** and missing data handling

### 🌊 **Flow Matching Model**
- **State-of-the-art flow matching** for performance dynamics
- **Conditional flow matching** for optimal transport learning
- **Neural ODE integration** for continuous trajectory modeling
- **Multi-dimensional performance** representation

### 📊 **Analysis and Evaluation**
- **Trajectory reconstruction** evaluation
- **Team performance interpolation** between different states
- **Comprehensive visualization** of model predictions
- **Statistical analysis** of flow dynamics

## Data Pipeline

The data generation pipeline produces team-week level observations with:

| Feature | Description |
|---------|-------------|
| `off_epa_per_play` | Offensive Expected Points Added per play |
| `success_rate` | Percentage of successful plays (0-1) |
| `pass_rate` | Percentage of passing plays (0-1) |
| `avg_air_yards` | Average air yards on passing attempts |
| `efficiency_score` | EPA × Success rate |
| `passing_efficiency` | Air yards × Pass rate |
| `total_plays` | Total offensive plays |

## Model Architecture

The flow matching model consists of:

- **Time Embedding Network**: Encodes temporal dynamics
- **Flow Network**: Neural network parameterizing velocity field
- **Conditional Flow Matching**: Learns optimal transport between performance states
- **Trajectory Sampling**: Generates realistic team performance paths

## Usage Examples

### Training

```python
from first_and_flow import NFLDataLoader, FlowMatchingModel, FlowTrainer

# Initialize components
data_loader = NFLDataLoader(data_path="../data/processed/nfl_flow_training_data.csv")
model = FlowMatchingModel(input_dim=7, hidden_dim=128)
trainer = FlowTrainer(model, data_loader)

# Train model
history = trainer.train(num_epochs=100)
```

### Analysis

```python
from first_and_flow import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(model, data_loader)

# Analyze team dynamics
analysis = evaluator.analyze_team_dynamics("KC", 2023)
evaluator.plot_team_analysis(analysis)

# Compare teams
comparison = evaluator.compare_teams("KC", "BUF", 2023)
evaluator.plot_team_comparison(comparison)
```

## Requirements

### R Environment (Data Generation)
- R >= 4.0.0
- nflfastR, dplyr, readr, stringr packages
- ~4GB RAM for multi-season processing
- Internet connection for data download

### Python Environment (Model Development)
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy, Pandas, Matplotlib, Seaborn
- CUDA optional but recommended for training

## Research Applications

This framework enables research in:
- **Team performance prediction** and trajectory forecasting
- **Optimal transport in sports analytics** 
- **Generative modeling of game dynamics**
- **Transfer learning between seasons/teams**
- **Counterfactual analysis** ("what if" scenarios)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{first_and_flow2025,
  title={First and Flow: Flow Matching for Football Analytics},
  author={First and Flow Team},
  year={2025},
  howpublished={\url{https://github.com/yourusername/first_and_flow}}
}
```
