"""
First and Flow: Flow Matching for Football

A machine learning framework for modeling NFL team performance dynamics
using flow matching techniques.
"""

__version__ = "0.1.0"
__author__ = "First and Flow Team"

from .data_loader import NFLDataLoader
from .flow_model import FlowMatchingModel
from .trainer import FlowTrainer
from .evaluator import ModelEvaluator

__all__ = [
    "NFLDataLoader",
    "FlowMatchingModel", 
    "FlowTrainer",
    "ModelEvaluator"
]
