#!/usr/bin/env python3
"""
First and Flow: Main Training Script

This script orchestrates the complete pipeline from data loading
to model training and evaluation for the NFL flow matching model.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from first_and_flow import NFLDataLoader, FlowMatchingModel, FlowTrainer, ModelEvaluator


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train and evaluate NFL Flow Matching Model')
    
    parser.add_argument('--data-path', type=str, 
                       default='../data/processed/nfl_flow_training_data.csv',
                       help='Path to processed NFL data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension of the model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Training device')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (requires trained model)')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting First and Flow pipeline...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = NFLDataLoader(
            data_path=args.data_path,
            normalize_features=True,
            sequence_length=4
        )
        
        # Check if data exists
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            logger.info("Please run the data generation pipeline first:")
            logger.info("cd ../data && Rscript generate_data.R")
            return 1
        
        # Load data to determine input dimensions
        data_loader.load_data()
        processed_data = data_loader.preprocess_features()
        
        # Get feature dimensions dynamically
        # Create a sample sequence to determine the number of features
        sample_sequences, _, _ = data_loader.create_sequences(processed_data.head(100))
        input_dim = sample_sequences.shape[-1]  # Get feature dimension
        
        logger.info(f"Detected input dimension: {input_dim}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = FlowMatchingModel(
            input_dim=input_dim,  # Use detected dimension
            hidden_dim=args.hidden_dim,
            num_layers=3,
            time_embedding_dim=32
        )
        
        if not args.eval_only:
            # Training phase
            logger.info("Starting training phase...")
            
            trainer = FlowTrainer(
                model=model,
                data_loader=data_loader,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                device=args.device,
                checkpoint_dir=args.checkpoint_dir
            )
            
            # Train the model
            history = trainer.train(
                num_epochs=args.epochs,
                save_every=10,
                early_stopping_patience=20
            )
            
            logger.info("Training completed successfully!")
            logger.info(f"Best validation loss: {min(history['val_losses']):.4f}")
            logger.info(f"Final test loss: {history['test_loss']:.4f}")
            
            # Save final results
            results_path = Path(args.checkpoint_dir) / "final_results.txt"
            with open(results_path, "w") as f:
                f.write("First and Flow Training Results\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Model Architecture:\n")
                f.write(f"  Input Dimension: {input_dim}\n")
                f.write(f"  Hidden Dimension: {args.hidden_dim}\n")
                f.write(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
                f.write(f"Training Configuration:\n")
                f.write(f"  Epochs: {args.epochs}\n")
                f.write(f"  Batch Size: {args.batch_size}\n")
                f.write(f"  Learning Rate: {args.learning_rate}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Best Validation Loss: {min(history['val_losses']):.4f}\n")
                f.write(f"  Final Test Loss: {history['test_loss']:.4f}\n")
            
            # Load best model for evaluation
            best_model_path = Path(args.checkpoint_dir) / "best_model.pt"
            if best_model_path.exists():
                trainer.load_checkpoint(str(best_model_path))
        
        else:
            # Evaluation only
            if not args.model_path:
                # Try to find best model in checkpoint directory
                best_model_path = Path(args.checkpoint_dir) / "best_model.pt"
                if best_model_path.exists():
                    args.model_path = str(best_model_path)
                else:
                    logger.error("No model path provided and no best model found")
                    return 1
            
            logger.info(f"Loading model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluation phase
        logger.info("Starting evaluation phase...")
        
        evaluator = ModelEvaluator(
            model=model,
            data_loader=data_loader,
            device=args.device
        )
        
        # Use the test data from the trainer's data split
        # Re-prepare data to get test sequences
        if not args.eval_only:
            # If we just trained, we can use the existing data loader state
            trainer_data_loader = trainer.data_loader
            trainer_data_loader.load_data() 
            processed_data = trainer_data_loader.preprocess_features()
            all_sequences, all_labels, all_names = trainer_data_loader.create_sequences(processed_data)
            
            # Use the same splitting logic as trainer
            seasons = processed_data['season'].unique()
            test_seasons = [seasons.max()]
            
            test_mask = []
            for i, name in enumerate(all_names):
                parts = name.split('_')
                season = int(parts[1])
                if season in test_seasons:
                    test_mask.append(i)
            
            test_sequences = all_sequences[test_mask]
        else:
            # For eval-only mode, create test sequences fresh
            data_loader.load_data()
            processed_data = data_loader.preprocess_features()
            _, _, test_df = data_loader.train_test_split()
            test_sequences, _, _ = data_loader.create_sequences(test_df)
        
        # Generate comprehensive evaluation report
        eval_dir = "evaluation_results"
        report = evaluator.generate_report(test_sequences, eval_dir)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {eval_dir}/")
        logger.info(f"Reconstruction error: {report['reconstruction_metrics']['mean_reconstruction_error']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
