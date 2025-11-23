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
    # Configuration
    task = "Lift-Panda"
    seed = 42
    buffer_size = 100000
    hidden_sizes = [256, 256]
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    auto_alpha = True
    epoch = 100
    step_per_epoch = 10000
    step_per_collect = 1
    update_per_step = 1
    batch_size = 256
    training_num = 1
    test_num = 5
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environments
    train_envs = DummyVectorEnv([make_robosuite_env for _ in range(training_num)])
    test_envs = DummyVectorEnv([make_robosuite_env for _ in range(test_num)])
    
    # Get environment dimensions
    env = make_robosuite_env()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    print(f"Observation shape: {state_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Action range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    env.close()
    
    # Create networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)
    
    # Setup alpha
    if auto_alpha:
        target_entropy = -np.prod(action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    
    # Create SAC policy
    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=1,
        action_space=env.action_space
    )
    
    # Create collectors
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=False
    )
    test_collector = Collector(policy, test_envs)
    
    # Warm up replay buffer
    print("Collecting initial random data...")
    train_collector.collect(n_step=batch_size * 10)
    
    # Training
    def save_best_fn(policy):
        torch.save(policy.state_dict(), f"{task}_sac_best.pth")
    
    def stop_fn(mean_rewards):
        return False  # Train for full epochs
    
    print(f"Training SAC on {task}...")
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn
    ).run()
    
    print(f"Training completed! Best reward: {result.best_reward}")
    
    # Save final model
    torch.save(policy.state_dict(), f"{task}_sac_final.pth")
    print(f"Final model saved to {task}_sac_final.pth")


if __name__ == "__main__":
    main()
