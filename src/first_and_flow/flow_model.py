"""
Flow Matching Model for NFL Team Performance

Implements flow matching techniques to model the dynamics of NFL team
performance over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FlowMatchingModel(nn.Module):
    """
    Flow Matching Model for NFL team performance dynamics.
    
    This model learns to transform between different team performance
    distributions using flow matching techniques.
    """
    
    def __init__(
        self,
        input_dim: int = 7,  # Number of performance features
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Initialize the Flow Matching Model.
        
        Args:
            input_dim: Dimension of input features (performance metrics)
            hidden_dim: Hidden layer dimension
            num_layers: Number of neural network layers
            time_embedding_dim: Dimension of time embeddings
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_embedder = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU()
        )
        
        # Main flow network
        layers = []
        current_dim = input_dim + time_embedding_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Output layer (velocity field)
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.flow_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the flow matching model.
        
        Args:
            x: Performance features [batch_size, input_dim]
            t: Time parameter [batch_size, 1] 
            
        Returns:
            Velocity field [batch_size, input_dim]
        """
        # Embed time parameter
        t_emb = self.time_embedder(t)
        
        # Concatenate features and time embedding
        xt = torch.cat([x, t_emb], dim=-1)
        
        # Compute velocity field
        velocity = self.flow_network(xt)
        
        return velocity
    
    def sample_trajectory(
        self,
        initial_state: torch.Tensor,
        num_steps: int = 50,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Sample a trajectory by integrating the flow.
        
        Args:
            initial_state: Starting performance state [input_dim]
            num_steps: Number of integration steps
            device: Device to run on
            
        Returns:
            Trajectory [num_steps, input_dim]
        """
        self.eval()
        
        with torch.no_grad():
            # Initialize trajectory
            trajectory = torch.zeros(num_steps, self.input_dim, device=device)
            trajectory[0] = initial_state
            
            dt = 1.0 / (num_steps - 1)
            
            for i in range(1, num_steps):
                t = torch.tensor([[i * dt]], device=device, dtype=torch.float32)
                x = trajectory[i-1:i]
                
                # Compute velocity
                velocity = self.forward(x, t)
                
                # Euler integration step
                trajectory[i] = trajectory[i-1] + dt * velocity.squeeze(0)
        
        return trajectory
    
    def interpolate_teams(
        self,
        team1_state: torch.Tensor,
        team2_state: torch.Tensor,
        num_steps: int = 20,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Interpolate between two team performance states.
        
        Args:
            team1_state: Performance state of first team
            team2_state: Performance state of second team
            num_steps: Number of interpolation steps
            device: Device to run on
            
        Returns:
            Interpolation path [num_steps, input_dim]
        """
        self.eval()
        
        with torch.no_grad():
            path = torch.zeros(num_steps, self.input_dim, device=device)
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                
                # Linear interpolation as target
                target = (1 - alpha) * team1_state + alpha * team2_state
                path[i] = target
        
        return path


class FlowLoss(nn.Module):
    """Loss function for flow matching training."""
    
    def __init__(self, sigma: float = 0.1):
        """
        Initialize flow matching loss.
        
        Args:
            sigma: Noise level for conditional flow matching
        """
        super().__init__()
        self.sigma = sigma
    
    def forward(
        self,
        velocity_pred: torch.Tensor,
        velocity_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            velocity_pred: Predicted velocity field
            velocity_target: Target velocity field
            
        Returns:
            Loss value
        """
        return F.mse_loss(velocity_pred, velocity_target)


class ConditionalFlowMatcher:
    """
    Conditional Flow Matching for NFL team performance.
    
    Implements the conditional flow matching algorithm for learning
    optimal transport between team performance distributions.
    """
    
    def __init__(self, sigma: float = 0.1):
        """
        Initialize conditional flow matcher.
        
        Args:
            sigma: Noise parameter for conditional paths
        """
        self.sigma = sigma
    
    def sample_conditional_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from conditional probability path.
        
        Args:
            x0: Source performance state
            x1: Target performance state  
            t: Time parameter
            
        Returns:
            Tuple of (sampled_state, target_velocity)
        """
        # Linear interpolation path
        mu_t = (1 - t) * x0 + t * x1
        
        # Add noise
        noise = torch.randn_like(mu_t) * self.sigma
        x_t = mu_t + noise
        
        # Target velocity (derivative of path)
        u_t = x1 - x0
        
        return x_t, u_t
    
    def generate_training_batch(
        self,
        team_sequences: torch.Tensor,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a training batch for flow matching.
        
        Args:
            team_sequences: Team performance sequences [N, seq_len, features]
            batch_size: Batch size
            
        Returns:
            Tuple of (x_t, t, target_velocity)
        """
        device = team_sequences.device
        N, seq_len, features = team_sequences.shape
        
        # Sample random pairs from sequences
        batch_x_t = []
        batch_t = []
        batch_u_t = []
        
        for _ in range(batch_size):
            # Sample random sequence
            seq_idx = torch.randint(0, N, (1,)).item()
            sequence = team_sequences[seq_idx]
            
            # Sample two random time points
            t1, t2 = torch.randint(0, seq_len, (2,))
            if t1 > t2:
                t1, t2 = t2, t1
            
            x0 = sequence[t1]
            x1 = sequence[t2]
            
            # Sample random time parameter
            t = torch.rand(1, device=device)
            
            # Generate conditional path sample
            x_t, u_t = self.sample_conditional_path(x0, x1, t)
            
            batch_x_t.append(x_t)
            batch_t.append(t)
            batch_u_t.append(u_t)
        
        return (
            torch.stack(batch_x_t),
            torch.stack(batch_t),
            torch.stack(batch_u_t)
        )


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = FlowMatchingModel(
        input_dim=7,
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    # Create dummy data
    batch_size = 16
    x = torch.randn(batch_size, 7, device=device)
    t = torch.rand(batch_size, 1, device=device)
    
    # Forward pass
    velocity = model(x, t)
    
    print(f"Model output shape: {velocity.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Sample trajectory
    initial_state = torch.randn(7, device=device)
    trajectory = model.sample_trajectory(initial_state, num_steps=20, device=device)
    print(f"Trajectory shape: {trajectory.shape}")
