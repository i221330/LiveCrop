# Colab Setup & Troubleshooting

This file documents how to run the LiveCrop training pipeline on Google Colab.

## Quick start

1. Open the notebook: **https://colab.research.google.com/github/i221330/LiveCrop/blob/main/notebooks/irrigation_rl.ipynb**
2. Run **Section 1** (install) — clones the repo and installs all dependencies
3. Run **Sections 2–7** sequentially to train, evaluate, and visualize
4. (Optional) Run **Section 8** to push results back to GitHub

## What the notebook does

| Section | Task | Time |
|---------|------|------|
| 1 | Clone repo + pip install | ~2 min |
| 2 | Run 20 episodes each: random + threshold baselines | ~1 min |
| 3 | Train DQN, PPO, A2C (200k steps each) | ~10 min |
| 4 | Evaluate all 3 agents on 30 held-out seeds | ~3 min |
| 5 | Generate comparison boxplot + trajectory plots | ~1 min |
| 6 | TensorBoard learning curves (interactive) | - |
| 7 | **Optional** seed sweep: train best algo × 5 seeds | ~20 min |
| 8 | **Optional** push plots/raw JSON to GitHub | ~10 sec |

**Total time (sections 1–6):** ~17 min on CPU, ~10 min on GPU.

## Dependencies & Version Pinning

Colab's pre-installed numpy often conflicts with gymnasium and SB3. To avoid the `ValueError: numpy.dtype size changed` error, `requirements.txt` pins:

```
numpy==1.26.4
gymnasium==0.29.1
stable-baselines3==2.3.0
```

These versions are tested to work together on Colab Python 3.10/3.12.

## Setup steps

### Step 1: Clone & install (automated in Section 1)

The notebook cell does this:

```python
import os
REPO_DIR = "LiveCrop"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/i221330/LiveCrop.git
os.chdir(REPO_DIR)
!pip install -q -e '.[train,plot,dev]'
```

If this fails, manually run in a cell:

```python
!git clone https://github.com/i221330/LiveCrop.git
%cd LiveCrop
!pip install -q -e '.[train,plot,dev]'
```

### Step 2: Restart kernel (if install fails)

If you see `ImportError` or `ValueError: numpy.dtype size changed` after install:

1. Click **Runtime** → **Restart session**
2. Re-run the install cell
3. Then proceed to imports

### Step 3: (Optional) Add GitHub token for Section 8

To push results back to GitHub from Colab:

1. Open **https://github.com/settings/tokens** in a new tab
2. Click **Generate new token** → **Generate new token (classic)**
3. Set:
   - **Note:** `Colab LiveCrop`
   - **Expiration:** 30 days
   - **Scopes:** Check `repo` only
4. **Copy the token** immediately
5. In Colab, go to left sidebar → 🔑 **Secrets**
6. Click **+ Add new secret**
7. **Name:** `GITHUB_TOKEN` (exact spelling)
8. **Value:** Paste the token
9. Toggle **Notebook access** to `ON`
10. Click **Add secret**

Section 8 will fetch it automatically:

```python
from google.colab import userdata
GH_TOKEN = userdata.get("GITHUB_TOKEN")
```

## Troubleshooting

### `ValueError: numpy.dtype size changed`

**Cause:** numpy version conflict between Colab's pre-installed version and what SB3 expects.

**Fix:**
1. Restart kernel: **Runtime** → **Restart session**
2. Re-run Section 1 install cell
3. If still fails, manually run:
   ```python
   !pip install --upgrade --force-reinstall numpy==1.26.4 gymnasium==0.29.1 stable-baselines3==2.3.0
   %cd LiveCrop  # if you left the dir
   !pip install -q -e '.[train,plot,dev]'
   ```

### Import errors after install

**Cause:** Kernel didn't pick up the new packages.

**Fix:** Restart the kernel (**Runtime** → **Restart session**) and re-run imports.

### Models take too long to train

The default is 200k timesteps (~10 min on CPU). To speed up for testing:

In **Section 3**, change:
```python
TIMESTEPS = 50_000   # quicker test run
```

For publication-quality results, use 500k steps (~25 min on CPU).

### Can't push to GitHub (Section 8 fails)

**Cause:** Missing or invalid `GITHUB_TOKEN` secret.

**Fix:**
1. Check the 🔑 **Secrets** panel — is `GITHUB_TOKEN` listed?
2. Is **Notebook access** toggled to `ON`?
3. Did you copy the full token from GitHub (no truncation)?
4. If the token expired, generate a new one at **https://github.com/settings/tokens**

If still failing, you can manually download plots:
1. Click the 📁 folder icon (Files) in the left sidebar
2. Right-click `results/plots/` → **Download**
3. Commit locally: `git add results/plots/ && git commit -m "Add training plots" && git push`

### Out of memory (OOM)

If training crashes with OOM:
1. Use **GPU** (**Runtime** → **Change runtime type** → select GPU)
2. Or reduce batch sizes in the SB3 hyperparameters (in `agents/train.py`)
3. Or reduce `TIMESTEPS` to 100k

## File structure after training

After running Sections 1–6, the local Colab filesystem contains:

```
LiveCrop/
  results/
    models/
      dqn_seed42.zip
      ppo_seed42.zip
      a2c_seed42.zip
    plots/
      baselines.png              # Week 1 baselines
      baselines_trajectory.png   # Week 1 baselines
      eval_comparison.png        # Week 3 RL vs baseline comparison
      trajectory_ppo.png         # Best algo vs threshold heuristic
    raw/
      eval_comparison.csv        # Episode-level results (all algos, all seeds)
      eval_summary.json          # Summary stats per policy
      sweep_ppo.json             # (if Section 7 ran)
    logs/
      dqn_seed42/                # TensorBoard event files
      ppo_seed42/
      a2c_seed42/
```

## Pushing results back to GitHub (Section 8)

Section 8 commits and pushes the `results/plots/` and `results/raw/` directories to `main`:

```python
!git add results/plots/ results/raw/
!git commit -m "Add Week 3 training results and portfolio plots [Colab]"
!git push origin main
```

After this runs, refresh **https://github.com/i221330/LiveCrop** and the plots will be live in the repo.

## Local alternative (if Colab is slow)

If you prefer to train locally (Mac/Linux/Windows):

```bash
pip install -r requirements.txt
pip install -e ".[train,plot,dev]"

python3 -m agents.train --algo ppo --timesteps 200000 --seed 42
python3 -m agents.train --algo dqn --timesteps 200000 --seed 42
python3 -m agents.train --algo a2c --timesteps 200000 --seed 42

python3 -m agents.evaluate --algos ppo dqn a2c
```

Results land in `results/plots/` and `results/raw/`.
