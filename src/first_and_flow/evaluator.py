"""
Model Evaluation and Analysis

Provides tools for evaluating trained flow matching models,
analyzing team performance dynamics, and generating insights.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from .flow_model import FlowMatchingModel
from .data_loader import NFLDataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluation and analysis tools for trained flow matching models.
    
    Provides functionality for:
    - Model performance evaluation
    - Team trajectory analysis
    - Performance interpolation visualization
    - Statistical analysis of flow dynamics
    """
    
    def __init__(
        self,
        model: FlowMatchingModel,
        data_loader: NFLDataLoader,
        device: str = "auto"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained flow matching model
            data_loader: NFL data loader
            device: Device for computation
        """
        self.model = model
        self.data_loader = data_loader
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate_reconstruction(
        self, 
        test_sequences: np.ndarray, 
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model's ability to reconstruct team trajectories.
        
        Args:
            test_sequences: Test sequences [N, seq_len, features]
            num_samples: Number of samples for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating trajectory reconstruction...")
        
        reconstruction_errors = []
        interpolation_errors = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(test_sequences))):
                sequence = torch.FloatTensor(test_sequences[i]).to(self.device)
                seq_len, features = sequence.shape
                
                # Sample start and end points
                start_idx = np.random.randint(0, seq_len - 1)
                end_idx = np.random.randint(start_idx + 1, seq_len)
                
                start_state = sequence[start_idx]
                end_state = sequence[end_idx]
                true_trajectory = sequence[start_idx:end_idx+1]
                
                # Generate model trajectory
                num_steps = end_idx - start_idx + 1
                model_trajectory = self.model.sample_trajectory(
                    start_state, num_steps=num_steps, device=self.device
                )
                
                # Compute reconstruction error
                reconstruction_error = torch.mean(
                    torch.norm(model_trajectory - true_trajectory, dim=1)
                ).item()
                reconstruction_errors.append(reconstruction_error)
                
                # Compute interpolation error (linear vs flow)
                linear_trajectory = torch.stack([
                    start_state + (end_state - start_state) * t / (num_steps - 1)
                    for t in range(num_steps)
                ])
                
                linear_error = torch.mean(
                    torch.norm(linear_trajectory - true_trajectory, dim=1)
                ).item()
                interpolation_errors.append(reconstruction_error - linear_error)
        
        metrics = {
            "mean_reconstruction_error": np.mean(reconstruction_errors),
            "std_reconstruction_error": np.std(reconstruction_errors),
            "mean_interpolation_improvement": np.mean(interpolation_errors),
            "reconstruction_errors": reconstruction_errors
        }
        
        logger.info(f"Reconstruction error: {metrics['mean_reconstruction_error']:.4f} ± {metrics['std_reconstruction_error']:.4f}")
        
        return metrics
    
    def analyze_team_dynamics(
        self, 
        team: str, 
        season: int,
        num_trajectory_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze the performance dynamics of a specific team.
        
        Args:
            team: Team abbreviation
            season: Season year
            num_trajectory_samples: Number of trajectory samples to generate
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing dynamics for {team} in {season}...")
        
        # Get actual team trajectory
        actual_trajectory = self.data_loader.get_team_trajectories(team, season)
        
        if len(actual_trajectory) == 0:
            logger.warning(f"No data found for {team} in {season}")
            return {}
        
        # Convert to tensor
        actual_tensor = torch.FloatTensor(actual_trajectory).to(self.device)
        
        # Generate model trajectories from different starting points
        sampled_trajectories = []
        
        with torch.no_grad():
            for i in range(min(num_trajectory_samples, len(actual_trajectory))):
                start_state = actual_tensor[i]
                
                # Sample trajectory of remaining season length
                remaining_weeks = len(actual_trajectory) - i
                if remaining_weeks > 1:
                    trajectory = self.model.sample_trajectory(
                        start_state, 
                        num_steps=remaining_weeks, 
                        device=self.device
                    )
                    sampled_trajectories.append(trajectory.cpu().numpy())
        
        return {
            "actual_trajectory": actual_trajectory,
            "sampled_trajectories": sampled_trajectories,
            "team": team,
            "season": season
        }
    
    def compare_teams(
        self, 
        team1: str, 
        team2: str, 
        season: int,
        num_interpolation_steps: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Compare two teams by interpolating between their performance states.
        
        Args:
            team1: First team abbreviation
            team2: Second team abbreviation
            season: Season year
            num_interpolation_steps: Number of interpolation steps
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {team1} vs {team2} in {season}...")
        
        # Get team trajectories
        team1_trajectory = self.data_loader.get_team_trajectories(team1, season)
        team2_trajectory = self.data_loader.get_team_trajectories(team2, season)
        
        if len(team1_trajectory) == 0 or len(team2_trajectory) == 0:
            logger.warning(f"Missing data for team comparison")
            return {}
        
        # Use season averages for comparison
        team1_avg = torch.FloatTensor(np.mean(team1_trajectory, axis=0)).to(self.device)
        team2_avg = torch.FloatTensor(np.mean(team2_trajectory, axis=0)).to(self.device)
        
        # Generate interpolation path
        with torch.no_grad():
            interpolation_path = self.model.interpolate_teams(
                team1_avg, team2_avg, 
                num_steps=num_interpolation_steps, 
                device=self.device
            )
        
        return {
            "team1": team1,
            "team2": team2,
            "team1_trajectory": team1_trajectory,
            "team2_trajectory": team2_trajectory,
            "team1_average": team1_avg.cpu().numpy(),
            "team2_average": team2_avg.cpu().numpy(),
            "interpolation_path": interpolation_path.cpu().numpy()
        }
    
    def plot_team_analysis(
        self, 
        analysis_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Plot team performance analysis results.
        
        Args:
            analysis_results: Results from analyze_team_dynamics
            save_path: Optional path to save the plot
        """
        if not analysis_results:
            logger.warning("No analysis results to plot")
            return
        
        actual = analysis_results["actual_trajectory"]
        sampled = analysis_results["sampled_trajectories"]
        team = analysis_results["team"]
        season = analysis_results["season"]
        
        feature_names = [
            'EPA/Play', 'Success Rate', 'Pass Rate',
            'Air Yards', 'Efficiency', 'Pass Efficiency', 'Total Plays'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names):
            ax = axes[i]
            
            # Plot actual trajectory
            weeks = range(1, len(actual) + 1)
            ax.plot(weeks, actual[:, i], 'o-', label='Actual', linewidth=2, markersize=6)
            
            # Plot sampled trajectories
            for j, traj in enumerate(sampled[:5]):  # Show first 5 samples
                start_week = j + 1
                traj_weeks = range(start_week, start_week + len(traj))
                ax.plot(traj_weeks, traj[:, i], '--', alpha=0.5, linewidth=1)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Week')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle(f'{team} Performance Dynamics - {season} Season', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_team_comparison(
        self, 
        comparison_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Plot team comparison results.
        
        Args:
            comparison_results: Results from compare_teams
            save_path: Optional path to save the plot
        """
        if not comparison_results:
            logger.warning("No comparison results to plot")
            return
        
        team1 = comparison_results["team1"]
        team2 = comparison_results["team2"]
        interpolation = comparison_results["interpolation_path"]
        
        feature_names = [
            'EPA/Play', 'Success Rate', 'Pass Rate',
            'Air Yards', 'Efficiency', 'Pass Efficiency', 'Total Plays'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names):
            ax = axes[i]
            
            # Plot interpolation path
            steps = range(len(interpolation))
            ax.plot(steps, interpolation[:, i], 'o-', linewidth=2, markersize=4)
            
            # Mark endpoints
            ax.scatter([0], [interpolation[0, i]], color='red', s=100, 
                      label=team1, zorder=5)
            ax.scatter([len(interpolation)-1], [interpolation[-1, i]], 
                      color='blue', s=100, label=team2, zorder=5)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Interpolation Step')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle(f'Performance Interpolation: {team1} → {team2}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(
        self,
        test_sequences: np.ndarray,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_sequences: Test data sequences
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Generating comprehensive evaluation report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Reconstruction evaluation
        reconstruction_metrics = self.evaluate_reconstruction(test_sequences)
        
        # Example team analysis (if data available)
        teams_to_analyze = ["KC", "BUF", "SF", "DAL"]  # Popular teams
        team_analyses = {}
        
        for team in teams_to_analyze:
            try:
                analysis = self.analyze_team_dynamics(team, 2023)
                if analysis:
                    team_analyses[team] = analysis
                    
                    # Save individual team plot
                    plot_path = output_path / f"{team}_dynamics_2023.png"
                    self.plot_team_analysis(analysis, str(plot_path))
                    
            except Exception as e:
                logger.warning(f"Could not analyze {team}: {e}")
        
        # Team comparisons
        comparisons = {}
        if len(team_analyses) >= 2:
            team_list = list(team_analyses.keys())
            comparison_key = f"{team_list[0]}_vs_{team_list[1]}"
            
            try:
                comparison = self.compare_teams(team_list[0], team_list[1], 2023)
                if comparison:
                    comparisons[comparison_key] = comparison
                    
                    # Save comparison plot
                    plot_path = output_path / f"{comparison_key}_comparison.png"
                    self.plot_team_comparison(comparison, str(plot_path))
                    
            except Exception as e:
                logger.warning(f"Could not compare teams: {e}")
        
        # Compile report
        report = {
            "reconstruction_metrics": reconstruction_metrics,
            "team_analyses": team_analyses,
            "team_comparisons": comparisons,
            "model_info": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "parameters": sum(p.numel() for p in self.model.parameters())
            }
        }
        
        # Save report summary
        summary_path = output_path / "evaluation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Flow Matching Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Reconstruction Metrics:\n")
            f.write(f"  Mean Error: {reconstruction_metrics['mean_reconstruction_error']:.4f}\n")
            f.write(f"  Std Error: {reconstruction_metrics['std_reconstruction_error']:.4f}\n")
            f.write(f"  Interpolation Improvement: {reconstruction_metrics['mean_interpolation_improvement']:.4f}\n\n")
            
            f.write(f"Teams Analyzed: {len(team_analyses)}\n")
            f.write(f"Team Comparisons: {len(comparisons)}\n\n")
            
            f.write("Model Information:\n")
            f.write(f"  Input Dimension: {report['model_info']['input_dim']}\n")
            f.write(f"  Hidden Dimension: {report['model_info']['hidden_dim']}\n")
            f.write(f"  Total Parameters: {report['model_info']['parameters']:,}\n")
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report


if __name__ == "__main__":
    # Example evaluation script
    from .data_loader import NFLDataLoader
    from .trainer import FlowTrainer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load trained model (assuming it exists)
    data_loader = NFLDataLoader()
    model = FlowMatchingModel(input_dim=7, hidden_dim=128)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, data_loader)
    
    # Generate sample test data
    data_loader.load_data()
    processed_data = data_loader.preprocess_features()
    _, _, test_df = data_loader.train_test_split()
    test_sequences, _, _ = data_loader.create_sequences(test_df)
    
    # Run evaluation
    report = evaluator.generate_report(test_sequences)
    
    print("Evaluation completed!")
    print(f"Reconstruction error: {report['reconstruction_metrics']['mean_reconstruction_error']:.4f}")
