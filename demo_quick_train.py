"""Quick demo to verify SAC training works with standard Robosuite Lift-Panda"""
import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import robosuite as suite


class RobosuiteWrapper(gym.Env):
    """Wrapper to make robosuite compatible with Gymnasium/Tianshou"""
    
    def __init__(self, env_name="Lift", robots="Panda", **kwargs):
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            **kwargs
        )
        
        # Define observation and action spaces
        obs_dict = self.env.reset()
        obs = self._flatten_obs(obs_dict)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        spec = self.env.action_spec
        self.action_space = gym.spaces.Box(
            low=spec[0], high=spec[1], dtype=np.float32
        )
    
    def _flatten_obs(self, obs_dict):
        """Flatten observation dictionary to numpy array"""
        obs_list = []
        for key in sorted(obs_dict.keys()):
            obs_list.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(obs_list).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        obs_dict = self.env.reset()
        obs = self._flatten_obs(obs_dict)
        return obs, {}
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = self._flatten_obs(obs_dict)
        return obs, reward, done, False, info
    
    def render(self):
        pass
    
    def close(self):
        self.env.close()


def make_robosuite_env():
    return RobosuiteWrapper(env_name="Lift", robots="Panda")


def main():
    print("=" * 60)
    print("Quick Demo: Robosuite Lift-Panda with Tianshou SAC")
    print("=" * 60)
    
    # Minimal configuration for quick test
    seed = 42
    buffer_size = 1000
    hidden_sizes = [128, 128]
    lr = 1e-3
    epoch = 2
    step_per_epoch = 500
    batch_size = 64
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environments
    train_envs = DummyVectorEnv([make_robosuite_env for _ in range(1)])
    test_envs = DummyVectorEnv([make_robosuite_env for _ in range(2)])
    
    env = make_robosuite_env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    
    print(f"\n✓ Environment: Lift-Panda (Standard Robosuite)")
    print(f"✓ Observation dim: {state_shape[0]}")
    print(f"✓ Action dim: {action_shape[0]}")
    print(f"✓ Action range: [{env.action_space.low[0]:.1f}, {max_action:.1f}]")
    env.close()
    
    # Create networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, 
                     device=device, unbounded=True, conditioned_sigma=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    
    net_c1 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, 
                concat=True, device=device)
    net_c2 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, 
                concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)
    
    target_entropy = -np.prod(action_shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=lr)
    alpha = (target_entropy, log_alpha, alpha_optim)
    
    policy = SACPolicy(
        actor=actor, actor_optim=actor_optim,
        critic1=critic1, critic1_optim=critic1_optim,
        critic2=critic2, critic2_optim=critic2_optim,
        tau=0.005, gamma=0.99, alpha=alpha,
        estimation_step=1, action_space=env.action_space
    )
    
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=False
    )
    test_collector = Collector(policy, test_envs)
    
    print(f"\n✓ Collecting initial data...")
    train_collector.collect(n_step=batch_size * 2)
    
    print(f"✓ Starting training ({epoch} epochs, {step_per_epoch} steps/epoch)...\n")
    
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=1,
        episode_per_test=2,
        batch_size=batch_size,
        update_per_step=1,
    ).run()
    
    print(f"\n{'=' * 60}")
    print(f"✓ Training completed successfully!")
    print(f"✓ Best test reward: {result.best_reward:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
