# Robosuite Lift Task with Tianshou SAC

This project uses the **standard Robosuite Lift environment** with the Panda robot and trains it using Tianshou's SAC (Soft Actor-Critic) algorithm.

## Environment
- **Task**: Lift (standard robosuite environment)
- **Robot**: Panda
- **Control**: Joint velocity control (default)
- **Observation**: Proprioceptive state (no camera)

## Files
- `train_sac_lift.py`: Main training script with SAC
- `test_env.py`: Test the robosuite environment setup

## Usage

### Test the environment:
```bash
python test_env.py
```

### Train SAC agent:
```bash
python train_sac_lift.py
```

## Configuration
The training script uses these default parameters:
- Buffer size: 100,000
- Network: 256x256 hidden layers
- Learning rates: 3e-4 (actor, critic, alpha)
- Batch size: 256
- Epochs: 100
- Steps per epoch: 10,000

You can modify these in the `main()` function of `train_sac_lift.py`.
