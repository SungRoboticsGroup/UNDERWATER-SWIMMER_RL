# Quick Start Guide

Get started with SALP in under 60 seconds! 🚀

## 🎬 Watch a Trained Agent (30 seconds)

The fastest way to see SALP in action:

```bash
# Automatically find and watch the best trained model
python quickstart/watch_agent.py
```

That's it! The script will auto-detect your best model and show the agent in action.

---

## 📚 Table of Contents

1. [Watch Trained Agents](#watch-trained-agents)
2. [List Available Models](#list-available-models)
3. [Interactive Manual Control](#interactive-manual-control)
4. [Zero-Config Examples](#zero-config-examples)
5. [Advanced Options](#advanced-options)
6. [Next Steps](#next-steps)

---

## 🤖 Watch Trained Agents

### Basic Usage

```bash
# Auto-detect best model (recommended for first-time users)
python quickstart/watch_agent.py

# Watch 10 episodes instead of default 5
python quickstart/watch_agent.py --episodes 10

# Use stochastic (exploratory) actions
python quickstart/watch_agent.py --stochastic
```

### Specify a Model

```bash
# Watch a specific model
python quickstart/watch_agent.py --model data/models/single_food_long_horizon_20251130_223921/best_model.zip

# You can use either .zip (SB3) or .pth (custom) models
python quickstart/watch_agent.py --model data/models/single_food_optimal_navigation/best_model.pth
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to model file (.zip or .pth) | Auto-detects best model |
| `--episodes` | Number of episodes to watch | 5 |
| `--stochastic` | Use exploratory actions | False (deterministic) |
| `--no-render` | Run without visualization | False (renders) |

---

## 📦 List Available Models

See all your trained models:

```bash
python quickstart/list_models.py
```

**Example output:**
```
======================================================================
📦 Available Trained Models
======================================================================

🤖 Stable Baselines3 Models (.zip):

  1. single_food_long_horizon_20251130_223921
     ⭐ best_model.zip               (  10.5 MB, Nov 30, 2025)
     
  2. sb3_sac_continuous_20251117_230050
     ⭐ best_model.zip               (   8.2 MB, Nov 17, 2025)

🔥 Custom PyTorch Models (.pth):

  1. single_food_optimal_navigation
     ⭐ best_model.pth               (   2.1 MB, Oct 16, 2025)
     
======================================================================
✓ Found 3 trained models
```

---

## 🎮 Interactive Manual Control

Want to understand the physics before watching the AI? Try manual control!

```bash
# Standalone robot physics demo
python scripts/utilities/salp_robot.py
```

### Controls:
- **HOLD SPACE** - Inhale water (slow contraction)
- **RELEASE SPACE** - Exhale water (slow expansion + thrust)
- **← / →** - Steer rear nozzle left/right
- **ESC** - Quit

This helps you understand:
- How jet propulsion works
- How nozzle steering affects movement
- What the AI agent is trying to learn

---

## 🎯 Zero-Config Examples

### Watch Best Model (Simplest!)

Perfect for complete beginners:

```bash
python quickstart/examples/watch_best_model.py
```

No configuration needed - just run and watch! This script:
- Finds your best trained model automatically
- Runs 3 quick episodes
- Shows performance statistics

---

## ⚙️ Advanced Options

### Watch Multiple Models in Sequence

```bash
# Create a simple bash script
for model in data/models/*/best_model.zip; do
    echo "Testing: $model"
    python quickstart/watch_agent.py --model "$model" --episodes 3
done
```

### Benchmark Performance (No Rendering)

```bash
# Fast performance testing without visualization
python quickstart/watch_agent.py --no-render --episodes 100
```

### Compare Deterministic vs Stochastic

```bash
# Deterministic (best actions)
python quickstart/watch_agent.py --episodes 10

# Stochastic (exploratory)
python quickstart/watch_agent.py --episodes 10 --stochastic
```

---

## 🎓 Next Steps

### Learn More
- 📖 [Main README](../README.md) - Full project documentation
- 🔬 [Research Background](../docs/SALP_RESEARCH.md) - Scientific details
- 🏋️ [Train Your Own Agent](../examples/demo.py) - Start training

### Train a New Model

```bash
# Interactive training (recommended)
python examples/demo.py train

# Continuous visual training
python examples/demo.py train --continuous

# Start from scratch with custom name
python examples/demo.py train --from-scratch --model-name my_experiment
```

### Modify and Experiment

Want to customize? The quick-start scripts are simple Python files you can modify:
- `watch_agent.py` - Main viewing script
- `list_models.py` - Model discovery
- `examples/` - Example scripts

---

## 🐛 Troubleshooting

### No Models Found

```bash
❌ No trained models found in data/models/
   Train a model first or specify a model path with --model
```

**Solution:** Train a model first:
```bash
python examples/demo.py train
```

### Import Errors

```bash
ModuleNotFoundError: No module named 'salp'
```

**Solution:** Install the package:
```bash
pip install -e .
```

### Pygame Issues

```bash
pygame.error: No available video device
```

**Solutions:**
1. Ensure you have a display available
2. Use `--no-render` for headless systems
3. Set up virtual display (Linux): `xvfb-run python quickstart/watch_agent.py`

---

## 📞 Getting Help

- **Issues?** Check the main [README.md](../README.md)
- **Questions?** Review the [examples/](../examples/) directory
- **Contributions?** We welcome improvements!

---

## 🎉 Quick Reference Card

```bash
# Most common commands
python quickstart/watch_agent.py              # Watch best model
python quickstart/list_models.py              # List all models
python quickstart/examples/watch_best_model.py  # Zero-config demo

# Manual control
python scripts/utilities/salp_robot.py        # Manual robot control

# Training (see main docs for more)
python examples/demo.py train                 # Train new model

# Advanced
python quickstart/watch_agent.py --model <path> --episodes 10 --stochastic --num-food 1
```

---

**Made with 🌊 by the GRASP Lab**

*Bio-inspired underwater robotics for everyone*
