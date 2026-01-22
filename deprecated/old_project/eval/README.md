# Navigation Evaluation System

Step-by-step guide to evaluate SALP robot navigation performance.

---

## 🚀 Complete Walkthrough

Follow these three steps to evaluate your trained navigation model:

### Step 1: Watch Single Navigation (Visual Verification) 👁️

**Purpose:** Visually confirm the agent can navigate before running 100 trials.

```bash
python eval/watch_navigation.py \
  --model data/models/single_food_long_horizon_20251130_223921/best_model.zip \
  --start 150,300 \
  --goal 650,300
```

**What you'll see:**
- Pygame window opens showing the environment
- Green circle = start position
- Food item = goal position
- Agent navigates from start to goal
- Progress printed every 100 steps

**Expected result:**
```
✅ SUCCESS! Goal reached in ~1800 steps!
   Final distance: 49.7px
```

**What to check:**
- Agent starts at the green circle (left side)
- Agent moves toward the food (goal, right side)
- Agent successfully reaches within 50px of goal
- Behavior looks smooth and purposeful

---

### Step 2: Collect Batch Data (Run 100 Trials) 📊

**Purpose:** Run many trials and save trajectory data for visualization.

```bash
python eval/collect_navigation_data.py \
  --model data/models/single_food_long_horizon_20251130_223921/best_model.zip \
  --trials 100 \
  --start 150,300 \
  --goal 650,300 \
  --output eval/results/navigation_data.pkl
```

**What happens:**
- Loads the model once
- Runs 100 navigation trials (no rendering, faster)
- Each trial starts at random orientation
- Records positions, actions, and metrics
- Saves everything to a `.pkl` file

**Progress output:**
```
Trial 1/100... ✓ SUCCESS | Steps: 1548 | Path ratio: 1.04 | Final dist: 49.7px
Trial 2/100... ✓ SUCCESS | Steps: 1813 | Path ratio: 1.20 | Final dist: 49.9px
...
Trial 100/100... ✓ SUCCESS | Steps: 1872 | Path ratio: 1.26 | Final dist: 50.0px
```

**Expected results:**
```
============================================================
DATA COLLECTION COMPLETE
============================================================
Success Rate: 98.0% (98/100)
Avg Path Length: 586.1 px
Avg Steps: 1757.9
============================================================

✓ Trajectory data saved to: eval/results/navigation_data.pkl
```

**Time:** Takes ~5-10 minutes for 100 trials

**Output file:** `eval/results/navigation_data.pkl` (can be reused!)

---

### Step 3: Generate Visualizations 📈

**Purpose:** Create three types of visualizations from the saved data.

```bash
python eval/visualize_navigation.py \
  --data eval/results/navigation_data.pkl
```

**What it generates:**

1. **Raw Trajectories** (`trajectories_*.png`)
   - Shows actual discrete position sequences
   - Each trial in different color (viridis)
   - See exact paths taken

2. **Spline Trajectories** (`splines_*.png`)
   - Ultra-smooth curved spline fits
   - Downsampled to ~20 control points per trajectory
   - High smoothing (s=500) for beautiful curves
   - Shows overall navigation patterns

3. **Heatmap** (`heatmap_*.png`)
   - GPS/traffic-style density visualization
   - Splines → 2D histogram → gaussian smoothing
   - Vibrant colormap (blue→cyan→yellow→red)
   - Shows path consistency and common corridors

**Expected output:**
```
============================================================
GENERATING VISUALIZATIONS
============================================================

Generating raw trajectories visualization...
✓ Saved to: eval/results/trajectories_20251208_190256.png

Generating spline trajectories visualization...
✓ Saved to: eval/results/splines_20251208_190256.png

Generating spline heatmap visualization...
✓ Saved to: eval/results/heatmap_20251208_190256.png

============================================================
✅ ALL VISUALIZATIONS COMPLETE!
============================================================
```

**Time:** Fast! ~10 seconds (no model loading or trials)

**Output files:** Three PNG images in `eval/results/`

---

## 📁 File Structure

```
eval/
├── watch_navigation.py          # Step 1: Watch single trial
├── collect_navigation_data.py   # Step 2: Collect batch data
├── visualize_navigation.py      # Step 3: Generate visualizations
└── results/
    ├── navigation_data.pkl      # Saved trajectory data (reusable!)
    ├── trajectories_*.png       # Raw trajectory visualization
    ├── splines_*.png            # Smooth spline visualization
    └── heatmap_*.png            # Density heatmap
```

---

## 📊 Metrics Computed

During data collection (Step 2), these metrics are calculated for each trial:

- **Success**: Did agent reach within 50px of goal?
- **Path Length**: Total distance traveled
- **Path Ratio**: Actual path length / optimal (straight line)
- **Straightness**: Optimal distance / path length
- **Lateral Deviation**: Mean perpendicular distance from straight line
- **Area Coverage**: Bounding box area explored
- **Steps**: Number of timesteps taken

---

## 🎨 Understanding the Visualizations

### 1. Raw Trajectories
- **What it shows:** Discrete position sequences as recorded
- **Use case:** See exact paths, identify outliers
- **Look for:** General direction, consistency, failures

### 2. Spline Trajectories
- **What it shows:** Smooth cubic spline fits to each path
- **Use case:** See overall navigation patterns without noise
- **Look for:** Curvature, smoothness, general flow

### 3. Heatmap
- **What it shows:** Density of where agents spend time
- **Use case:** Find common corridors, bottlenecks
- **Look for:** 
  - Hot spots (red/yellow) = frequently visited
  - Cold spots (blue) = rarely visited
  - Narrow corridors = consistent paths

---

## 🔧 Customization

### Different Model
Use any trained `.zip` model:
```bash
python eval/watch_navigation.py \
  --model data/models/YOUR_MODEL/best_model.zip
```

### Different Start/Goal Positions
```bash
python eval/collect_navigation_data.py \
  --model YOUR_MODEL.zip \
  --start 100,100 \
  --goal 700,500
```

### More Trials (for publication)
```bash
python eval/collect_navigation_data.py \
  --trials 500 \
  --model YOUR_MODEL.zip
```

### Regenerate Visualizations (different style)
Since data is saved, you can regenerate visualizations anytime:
```bash
python eval/visualize_navigation.py \
  --data eval/results/navigation_data.pkl \
  --output-dir eval/results/v2
```

---

## ⚠️ Troubleshooting

### "IsADirectoryError"
- Need `.zip` file, not directory
- Find models: `find data/models -name "*.zip"`

### "Model takes too long / doesn't reach goal"
- Try different model (check training results)
- Increase `--max-steps` (default: 3000)

### "Visualizations look weird"
- Check success rate in Step 2 (should be >90%)
- Verify start/goal positions make sense
- Try with fewer trials first (--trials 10)

---

## 📝 Example Session

```bash
# Terminal 1: Watch one trial
python eval/watch_navigation.py \
  --model data/models/single_food_long_horizon_20251130_223921/best_model.zip \
  --start 150,300 --goal 650,300
# ✅ SUCCESS! Goal reached in 1859 steps!

# Terminal 2: Collect 100 trials
python eval/collect_navigation_data.py \
  --model data/models/single_food_long_horizon_20251130_223921/best_model.zip \
  --trials 100 \
  --start 150,300 --goal 650,300 \
  --output eval/results/navigation_data.pkl
# ✅ Success Rate: 98.0% (98/100)

# Terminal 3: Generate visualizations
python eval/visualize_navigation.py \
  --data eval/results/navigation_data.pkl
# ✅ ALL VISUALIZATIONS COMPLETE!

# View results
open eval/results/trajectories_*.png
open eval/results/splines_*.png
open eval/results/heatmap_*.png
```

**Done!** You now have quantitative metrics and beautiful visualizations.

---

## 💡 Why This Workflow?

✅ **Visual verification first** - Catch issues before running 100 trials

✅ **Efficient** - Collect data once, visualize many times

✅ **Reproducible** - Save data for later analysis

✅ **Shareable** - Send `.pkl` files to collaborators

✅ **Clean** - Each step is independent and focused

---

## 🎯 Quick Reference

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1. Watch | `watch_navigation.py` | 1 min | Visual confirmation |
| 2. Collect | `collect_navigation_data.py` | 5-10 min | `.pkl` data file |
| 3. Visualize | `visualize_navigation.py` | 10 sec | 3 PNG images |

**Total time:** ~10 minutes to go from model to publication-ready figures!
