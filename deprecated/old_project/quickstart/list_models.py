#!/usr/bin/env python3
"""
List all available trained SALP models.

Usage:
    python quickstart/list_models.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB"""
    return path.stat().st_size / (1024 * 1024)


def get_file_date(path: Path) -> str:
    """Get file modification date"""
    timestamp = path.stat().st_mtime
    return datetime.fromtimestamp(timestamp).strftime("%b %d, %Y")


def list_models():
    """List all available trained models"""
    models_dir = project_root / "data" / "models"
    
    if not models_dir.exists():
        print("❌ No models directory found at data/models/")
        print("   Train a model first!")
        return
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("❌ No model directories found in data/models/")
        print("   Train a model first!")
        return
    
    print(f"\n{'='*70}")
    print("📦 Available Trained Models")
    print(f"{'='*70}\n")
    
    # Categorize models
    sb3_models = []
    custom_models = []
    
    for model_dir in sorted(model_dirs):
        # Look for model files
        zip_files = list(model_dir.glob("*.zip"))
        pth_files = list(model_dir.glob("*.pth"))
        
        if zip_files or pth_files:
            model_info = {
                'name': model_dir.name,
                'path': model_dir,
                'zip_files': zip_files,
                'pth_files': pth_files,
            }
            
            if zip_files:
                sb3_models.append(model_info)
            if pth_files:
                custom_models.append(model_info)
    
    # Display SB3 models
    if sb3_models:
        print("🤖 Stable Baselines3 Models (.zip):")
        print()
        
        for i, model_info in enumerate(sb3_models, 1):
            print(f"  {i}. {model_info['name']}")
            
            # Show available model files
            for zip_file in sorted(model_info['zip_files']):
                size_mb = get_file_size_mb(zip_file)
                date = get_file_date(zip_file)
                marker = "⭐" if zip_file.name == "best_model.zip" else "  "
                print(f"     {marker} {zip_file.name:30s} ({size_mb:>6.1f} MB, {date})")
            
            print()
    
    # Display custom models
    if custom_models:
        print("🔥 Custom PyTorch Models (.pth):")
        print()
        
        for i, model_info in enumerate(custom_models, 1):
            print(f"  {i}. {model_info['name']}")
            
            # Show available model files
            for pth_file in sorted(model_info['pth_files']):
                # Skip training state files
                if 'training_state' in pth_file.name:
                    continue
                if 'discriminator' in pth_file.name:
                    continue
                    
                size_mb = get_file_size_mb(pth_file)
                date = get_file_date(pth_file)
                marker = "⭐" if pth_file.name == "best_model.pth" else "  "
                print(f"     {marker} {pth_file.name:30s} ({size_mb:>6.1f} MB, {date})")
            
            print()
    
    # Show usage instructions
    print(f"{'='*70}")
    print("📖 How to Watch a Model:")
    print(f"{'='*70}\n")
    
    print("Option 1 - Auto-select best model:")
    print("  python quickstart/watch_agent.py\n")
    
    print("Option 2 - Specify a model:")
    if sb3_models:
        example = sb3_models[0]
        best_model = next((f for f in example['zip_files'] if f.name == 'best_model.zip'), example['zip_files'][0])
        print(f"  python quickstart/watch_agent.py --model {best_model.relative_to(project_root)}\n")
    elif custom_models:
        example = custom_models[0]
        best_model = next((f for f in example['pth_files'] if f.name == 'best_model.pth'), example['pth_files'][0])
        print(f"  python quickstart/watch_agent.py --model {best_model.relative_to(project_root)}\n")
    
    print("Option 3 - Watch multiple episodes:")
    print("  python quickstart/watch_agent.py --episodes 10\n")
    
    print(f"{'='*70}\n")
    
    # Summary
    total_models = len(sb3_models) + len(custom_models)
    print(f"✓ Found {total_models} trained model{'s' if total_models != 1 else ''}")
    print()


def main():
    list_models()


if __name__ == "__main__":
    main()
