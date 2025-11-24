import gymnasium as gym
import numpy as np
import torch
from train_sac_lift import make_robosuite_env
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Batch

def evaluate_model(model_path, num_episodes=10):
    """Evaluate a trained SAC model"""
    
    # Create environment
    env = make_robosuite_env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    
    # Create networks (same as training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = [256, 256]
    
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True
    ).to(device)
    
    net_c1 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, concat=True, device=device)
    net_c2 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic2 = Critic(net_c2, device=device).to(device)
    
    # Create policy
    policy = SACPolicy(
        actor=actor,
        actor_optim=None,
        critic1=critic1,
        critic1_optim=None,
        critic2=critic2,
        critic2_optim=None,
        action_space=env.action_space
    )
    
    # Load trained weights
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Evaluating model: {model_path}")
    print(f"Running {num_episodes} episodes...\n")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 500:  # Max 500 steps per episode
            with torch.no_grad():
                batch = Batch(obs=np.array([obs]), info={})
                action_result = policy(batch, state=None)
                action = action_result.act[0]
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Check success (if info contains success flag)
        if 'success' in info and info['success']:
            success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward={episode_reward:.3f}, Length={steps}")
    
    env.close()
    
    # Summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Min Reward: {np.min(episode_rewards):.3f}")
    print(f"Max Reward: {np.max(episode_rewards):.3f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    if success_count > 0:
        print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print("="*50)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_count / num_episodes if num_episodes > 0 else 0
    }

if __name__ == "__main__":
    model_path = "Lift-Panda_sac_best.pth"
    results = evaluate_model(model_path, num_episodes=20)
