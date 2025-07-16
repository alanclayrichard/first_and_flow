"""
Training Module for Flow Matching Model

Handles training loop, optimization, and model checkpointing for the
NFL team performance flow matching model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

from .flow_model import FlowMatchingModel, FlowLoss, ConditionalFlowMatcher
from .data_loader import NFLDataLoader

logger = logging.getLogger(__name__)


class NFLSequenceDataset(Dataset):
    """Dataset class for NFL team performance sequences."""
    
    def __init__(self, sequences: np.ndarray, team_labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Team performance sequences [N, seq_len, features]
            team_labels: Team labels [N]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.team_labels = team_labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.team_labels[idx]


class FlowTrainer:
    """
    Trainer class for the Flow Matching Model.
    
    Handles the complete training pipeline including:
    - Data loading and batching
    - Loss computation using conditional flow matching
    - Optimization and learning rate scheduling
    - Model checkpointing and evaluation
    """
    
    def __init__(
        self,
        model: FlowMatchingModel,
        data_loader: NFLDataLoader,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Flow matching model to train
            data_loader: NFL data loader
            learning_rate: Initial learning rate
            batch_size: Training batch size
            device: Training device ("auto", "cpu", or "cuda")
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup training components
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.loss_fn = FlowLoss()
        self.flow_matcher = ConditionalFlowMatcher()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare training, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Preparing data for training...")
        
        # Load and preprocess ALL data first
        self.data_loader.load_data()
        processed_data = self.data_loader.preprocess_features()
        
        # Create sequences from the fully processed data
        all_sequences, all_labels, all_names = self.data_loader.create_sequences(processed_data)
        
        # Now split the sequences (instead of splitting raw data first)
        logger.info("Splitting sequences into train/val/test...")
        
        # Get unique season info for temporal splitting
        seasons = processed_data['season'].unique()
        test_seasons = [seasons.max()]  # Use latest season for testing
        
        # Create masks for splitting sequences based on team-season info
        train_mask = []
        val_mask = []
        test_mask = []
        
        np.random.seed(42)
        teams = processed_data['team'].unique()
        val_teams = np.random.choice(teams, size=int(len(teams) * 0.2), replace=False)
        
        for i, name in enumerate(all_names):
            # Parse team and season from sequence name (format: "TEAM_SEASON_weekN")
            parts = name.split('_')
            team = parts[0]
            season = int(parts[1])
            
            if season in test_seasons:
                test_mask.append(i)
            elif team in val_teams:
                val_mask.append(i)
            else:
                train_mask.append(i)
        
        # Split sequences
        train_sequences = all_sequences[train_mask]
        train_labels = all_labels[train_mask]
        
        val_sequences = all_sequences[val_mask] 
        val_labels = all_labels[val_mask]
        
        test_sequences = all_sequences[test_mask]
        test_labels = all_labels[test_mask]
        
        # Create datasets and loaders
        train_dataset = NFLSequenceDataset(train_sequences, train_labels)
        val_dataset = NFLSequenceDataset(val_sequences, val_labels)
        test_dataset = NFLSequenceDataset(test_sequences, test_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(data_loader, desc="Training")
        
        for sequences, _ in pbar:
            sequences = sequences.to(self.device)
            
            # Generate training batch using conditional flow matching
            x_t, t, target_velocity = self.flow_matcher.generate_training_batch(
                sequences, batch_size=min(self.batch_size, len(sequences))
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_velocity = self.model(x_t, t)
            
            # Compute loss
            loss = self.loss_fn(predicted_velocity, target_velocity)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def validate_epoch(self, data_loader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, _ in data_loader:
                sequences = sequences.to(self.device)
                
                # Generate validation batch
                x_t, t, target_velocity = self.flow_matcher.generate_training_batch(
                    sequences, batch_size=min(self.batch_size, len(sequences))
                )
                
                # Forward pass
                predicted_velocity = self.model(x_t, t)
                
                # Compute loss
                loss = self.loss_fn(predicted_velocity, target_velocity)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        num_epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train the flow matching model.
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch + 1, val_loss, is_best=True)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        test_loss = self.validate_epoch(test_loader)
        logger.info(f"Final test loss: {test_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "test_loss": test_loss
        }
    
    def save_checkpoint(
        self, 
        epoch: int, 
        val_loss: float, 
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]


if __name__ == "__main__":
    # Example training script
    from .data_loader import NFLDataLoader
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    data_loader = NFLDataLoader()
    model = FlowMatchingModel(input_dim=7, hidden_dim=128)
    trainer = FlowTrainer(model, data_loader, learning_rate=1e-3)
    
    # Train model
    history = trainer.train(num_epochs=50)
    
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_losses']):.4f}")
