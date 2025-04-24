#!/usr/bin/env python
"""
Model Registry CLI Tool
Provides command-line tools for managing the model registry
"""
import os
import sys
import argparse
import json
from typing import List, Dict, Any

from model_registry import list_models, get_model_details, delete_model

def print_model_list(models: List[Dict[str, Any]]):
    """Print list of models in a formatted way"""
    if not models:
        print("No models found in registry.")
        return
    
    # Print header
    print(f"{'MODEL ID':<40} {'MODEL TYPE':<15} {'DATASET':<10} {'ACCURACY':<10} {'STATUS':<12} {'CREATED AT':<25}")
    print("-" * 112)
    
    # Print each model
    for model in models:
        accuracy = f"{model.get('accuracy', 0):.4f}" if model.get('accuracy') is not None else "N/A"
        print(f"{model['model_id']:<40} {model['model_type']:<15} {model['dataset']:<10} {accuracy:<10} {model['status']:<12} {model['created_at'][:19]:<25}")

def print_model_details(model: Dict[str, Any]):
    """Print detailed information about a model"""
    if not model:
        print("Model not found.")
        return
    
    print(f"Model ID:       {model['model_id']}")
    print(f"Model Type:     {model['model_type']}")
    print(f"Dataset:        {model['dataset']}")
    print(f"Status:         {model['status']}")
    print(f"Created At:     {model['created_at']}")
    print(f"Job ID:         {model.get('training_job_id', 'N/A')}")
    print(f"Path:           {model.get('path', 'N/A')}")
    
    if model.get('description'):
        print(f"\nDescription:    {model['description']}")
    
    # Print metrics if available
    if model.get('metrics'):
        print("\nMetrics:")
        metrics = model['metrics']
        
        if 'best_val_acc' in metrics:
            print(f"  Best Validation Accuracy:  {metrics['best_val_acc']:.4f}")
        
        if 'val_acc' in metrics and isinstance(metrics['val_acc'], list) and metrics['val_acc']:
            print(f"  Final Validation Accuracy: {metrics['val_acc'][-1]:.4f}")
        
        if 'training_time' in metrics:
            minutes = int(metrics['training_time'] // 60)
            seconds = int(metrics['training_time'] % 60)
            print(f"  Training Time:            {minutes}m {seconds}s")
        
        if 'epochs' in metrics:
            print(f"  Epochs:                   {metrics['epochs']}")
        
        if 'batch_size' in metrics:
            print(f"  Batch Size:               {metrics['batch_size']}")
        
        if 'learning_rate' in metrics:
            print(f"  Learning Rate:            {metrics['learning_rate']}")
        
        if 'total_params' in metrics:
            print(f"  Total Parameters:         {metrics['total_params']:,}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Model Registry CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List all models in the registry")
    
    # Show model details command
    show_parser = subparsers.add_parser("show", help="Show details for a specific model")
    show_parser.add_argument("model_id", help="ID of the model to show")
    
    # Delete model command
    delete_parser = subparsers.add_parser("delete", help="Delete a model from the registry")
    delete_parser.add_argument("model_id", help="ID of the model to delete")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "list":
        models = list_models()
        print_model_list(models)
    
    elif args.command == "show":
        model = get_model_details(args.model_id)
        if model:
            print_model_details(model)
        else:
            print(f"Model with ID '{args.model_id}' not found.")
            sys.exit(1)
    
    elif args.command == "delete":
        success = delete_model(args.model_id)
        if success:
            print(f"Model with ID '{args.model_id}' successfully deleted.")
        else:
            print(f"Failed to delete model with ID '{args.model_id}'.")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 