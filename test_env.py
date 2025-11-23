import robosuite as suite
import numpy as np

# Test standard Lift environment with Panda robot
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

print("Testing Robosuite Lift-Panda environment...")
print(f"Action spec: {env.action_spec}")
print(f"Action dim: {env.action_dim}")

obs = env.reset()
print(f"Observation type: {type(obs)}")
if isinstance(obs, dict):
    print("Observation keys:", obs.keys())
    total_dim = sum(np.array(v).size for v in obs.values())
    print(f"Total observation dimension: {total_dim}")
else:
    print(f"Observation shape: {obs.shape}")

# Test a few random steps
for i in range(5):
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.4f}, done={done}")

env.close()
print("Environment test successful!")
