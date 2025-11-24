# Robosuite Lift Task with Tianshou SAC

This project uses the **standard Robosuite Lift environment** with the Panda robot and trains it using Tianshou's SAC (Soft Actor-Critic) algorithm.

## Environment
- **Task**: Lift (standard robosuite environment)
- **Robot**: Panda
- **Control**: Joint velocity control (default)
- **Observation**: Proprioceptive state (no camera)
- **Reward**: Dense rewards (`reward_shaping=True`) for better learning

## Files
- `train_sac_lift.py`: Main training script with SAC
- `evaluate_model.py`: Evaluate trained model
- `test_env.py`: Test the robosuite environment setup
- `Lift-Panda_sac_best.pth`: Best trained model checkpoint
- `Lift-Panda_sac_best.pth.backup`: Backup of previous model

## Usage

### Test the environment:
```bash
python test_env.py
```

### Train SAC agent:
```bash
python train_sac_lift.py
```

### Evaluate trained model:
```bash
python evaluate_model.py
```

## Configuration
The training script uses these default parameters:
- Buffer size: 100,000
- Network: 256x256 hidden layers
- Learning rates: 3e-4 (actor, critic, alpha)
- Batch size: 256
- Epochs: 100
- Steps per epoch: 10,000
- **Reward shaping: Enabled** (provides dense rewards for learning)

You can modify these in the `main()` function of `train_sac_lift.py`.

## Important Notes

⚠️ **Reward Shaping**: This project uses `reward_shaping=True` in the environment configuration. This provides dense rewards that give continuous feedback during training (reaching, grasping, lifting), making it feasible for RL algorithms to learn. Without reward shaping, the environment only returns reward=1 on success and 0 otherwise, which is extremely difficult to learn from scratch.

Expected rewards during training:
- Early epochs: 0.5 - 2.0
- Mid training: 2.0 - 10.0
- Successful task completion: 10.0+

## Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Flow                       │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   Initialize     │
    │   Environment    │
    │  (Robosuite)     │
    │   Lift-Panda     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Create SAC     │
    │    Agent with    │
    │  Actor-Critic    │
    │    Networks      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Initialize     │
    │  Replay Buffer   │
    │   (100k size)    │
    └────────┬─────────┘
             │
             ▼
    ╔══════════════════╗
    ║  Training Loop   ║
    ║  (100 epochs)    ║
    ╚════════┬═════════╝
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐  ┌──────────┐
│ Collect  │  │  Update  │
│Experience│─▶│  Policy  │
│(10k steps│  │  (SAC)   │
└──────────┘  └─────┬────┘
      ▲             │
      │             │
      └─────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Save Best Model │
    │  (if improved)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Evaluation     │
    │  (10 episodes)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Trained Model   │
    │  (.pth file)     │
    └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Environment Interaction                      │
└─────────────────────────────────────────────────────────────────┘

    State (Robot & Cube)
           │
           ▼
    ┌──────────────────┐
    │   Actor Network  │
    │  (Policy: π)     │
    └────────┬─────────┘
             │
             ▼
    Action (Joint Velocities)
             │
             ▼
    ┌──────────────────┐
    │  Robosuite Env   │
    │   Executes       │
    └────────┬─────────┘
             │
             ▼
    Next State + Reward + Done
             │
             ▼
    ┌──────────────────┐
    │  Replay Buffer   │
    │  (Store tuple)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Critic Networks  │
    │ (Q1, Q2) + α     │
    │  Learning Step   │
    └──────────────────┘
```
