# Expert Demonstrations for GAIL

This directory contains expert demonstrations used for GAIL (Generative Adversarial Imitation Learning) training.

## Directory Structure

```
expert_demos/
├── human/          # Human demonstrations (manual gameplay)
├── agent/          # Agent demonstrations (from trained models)
└── README.md       # This file
```

## Collecting Demonstrations

### Human Demonstrations

Collect demonstrations by playing the environment manually:

```bash
python scripts/collect_human_demos.py
```

**Controls:**
- Arrow Keys: Steer nozzle left/right
- Space: Inhale/Exhale (if forced_breathing is disabled)
- After episode ends:
  - Y: Save episode
  - N: Discard episode
  - Q: Quit collection

### Agent Demonstrations

Collect demonstrations from a trained SAC agent:

```bash
python scripts/collect_agent_demos.py --model models/path/to/best_model.pth
```

**Options:**
- `--num-episodes`: Number of episodes to collect (default: 50)
- `--min-score`: Minimum score threshold
- `--top-k`: Save only top K episodes
- `--stochastic`: Use stochastic policy (default: deterministic)
- `--render`: Visualize collection process

**Example:**
```bash
# Collect top 20 episodes from 50 attempts, with minimum score of 100
python scripts/collect_agent_demos.py \
    --model models/best_model.pth \
    --num-episodes 50 \
    --top-k 20 \
    --min-score 100.0
```

## Demonstration Format

Each demonstration file (`.pkl`) contains:
```python
{
    'observations': np.array([T, obs_dim]),
    'actions': np.array([T, action_dim]),
    'rewards': np.array([T]),
    'next_observations': np.array([T, obs_dim]),
    'dones': np.array([T]),
    'metadata': {
        'source': 'human' or 'agent',
        'score': float,
        'timestamp': str,
        'episode_length': int,
        'food_collected': int
    }
}
```

## Best Practices

1. **Quality over Quantity**: 10 high-quality demonstrations are better than 100 mediocre ones
2. **Diverse Strategies**: Collect demonstrations with different approaches
3. **Mix Sources**: Use both human intuition and agent performance
4. **Score Threshold**: Filter for successful episodes (e.g., score >= 50)
5. **Balance**: Aim for ~30% human, ~70% agent demonstrations

## Recommended Collection Strategy

### Phase 1: Bootstrap with Human Demos
```bash
# Collect 5-10 human demonstrations
python scripts/collect_human_demos.py
# Target: Good coverage of basic strategies
```

### Phase 2: Train Initial SAC Agent
```bash
# Train baseline SAC agent without GAIL
python demo.py  # or your training script
```

### Phase 3: Collect Agent Demos
```bash
# Collect top-performing episodes from trained agent
python scripts/collect_agent_demos.py \
    --model models/run_name/best_model.pth \
    --num-episodes 100 \
    --top-k 30 \
    --min-score 100.0
```

### Phase 4: Train with GAIL
```bash
# Train new agent with GAIL using combined demonstrations
python scripts/train_sac_gail.py
```

## Verification

Check your expert buffer statistics:

```python
from training.expert_buffer import ExpertBuffer

# Load buffer
buffer = ExpertBuffer(obs_dim=26, action_dim=1)  # Adjust dimensions
buffer.load_directory('expert_demos/')

# Print statistics
stats = buffer.get_statistics()
print(stats)
```

## Tips

- Start with at least 5 demonstrations
- More demonstrations = better GAIL performance
- High-performing demos are crucial
- Delete poor-quality demos manually if needed
- Back up your demos before major changes
