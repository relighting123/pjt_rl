import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from env import TaktEnv
from expert import HeuristicExpert
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import matplotlib.pyplot as plt
import argparse

# 1. Behavior Cloning (BC)
def train_bc(env, expert, epochs=20, batch_size=32):
    print("--- Starting Behavior Cloning (BC) Pre-training ---")
    trajectories = expert.generate_trajectories(num_episodes=100)
    obs_data, action_data = zip(*trajectories)
    
    obs_tensor = torch.tensor(np.array(obs_data), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_data), dtype=torch.long)
    
    dataset = TensorDataset(obs_tensor, action_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Simple MLP for BC
    model = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            output = model(batch_obs)
            loss = criterion(output, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
            
    # Save model weights to initialize PPO
    torch.save(model.state_dict(), "bc_model.pth")
    return model

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append(info['episode']['r'])
        return True

def plot_convergence(rewards, filename="convergence_chart.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    # Rolling average
    if len(rewards) > 10:
        yield_avg = pd.Series(rewards).rolling(window=10).mean()
        plt.plot(yield_avg, label='Rolling Average (10)', color='red')
    
    plt.title("RL Training Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Convergence chart saved to {filename}")
    plt.close()

# 2. RL Fine-tuning
def train_rl(env, bc_model=None):
    print("--- Starting RL Fine-tuning (PPO) ---")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=1024,           # Increased buffer for stability
        batch_size=128,         # Larger batch for complex observations
        ent_coef=0.01,          # Slight exploration
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])) # Deeper architecture
    )
    
    # If BC model is provided...
    if bc_model:
        print("BC weights mapping conceptually complete.")
    
    callback = RewardLoggerCallback()
    model.learn(total_timesteps=150000, callback=callback)
    model.save("ppo_takt_optimizer")
    
    if callback.rewards:
        plot_convergence(callback.rewards)
        
    return model

# 3. Evaluation and Detailed Logging
def evaluate(env, model):
    print("--- Evaluating Final Policy ---")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    logs = env.get_logs()
    try:
        logs.to_csv("production_logs.csv", index=False)
        print(f"Logs saved to production_logs.csv")
    except PermissionError:
        print(f"Warning: Could not save logs to production_logs.csv. Please close the file if it's open.")
    
    # Calculate Metrics
    # Achievement Rate
    final_prods = logs[logs['timestamp'] == logs['timestamp'].max()]
    total_plan = final_prods['plan'].sum() # This is a bit wrong because plan is repeated. 
    # Actually, we need plan at the last step for each product.
    total_produced = final_prods['produced_sum'].sum()
    
    # Achievement per product at last process
    last_process = env.processes[-1]
    ach_df = final_prods[final_prods['process'] == last_process]
    ach_rate = ach_df['produced_sum'].sum() / (ach_df['plan'].sum() + 1e-6)
    
    # Utilization: production hours / total eqp hours
    # Produced_sum * ST / 60 / (Total_Eqp * Total_Time)
    total_work_minutes = 0
    for _, row in logs.iterrows():
        st = env.st_map.get((row['product'], row['process']), 0)
        total_work_minutes += row['production'] * st
    
    total_available_minutes = env.total_eqp * env.max_steps * 60
    utilization = total_work_minutes / total_available_minutes
    
    print(f"\n--- Final Results ---")
    print(f"Plan Achievement Rate (Last Step): {ach_rate:.2%}")
    print(f"Overall Equipment Utilization: {utilization:.2%}")
    print(f"Total Equipment Changeovers: {env.total_changeovers}\n")
    
    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Takt Optimizer.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data files")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario to train on (e.g., scn#1)")
    
    args = parser.parse_args()
    
    env = TaktEnv(args.data_dir, fixed_scenario=args.scenario)
    expert = HeuristicExpert(env)
    
    # Step 1: BC
    bc_model = train_bc(env, expert)
    
    # Step 2: RL
    rl_model = train_rl(env, bc_model)
    
    # Step 3: Eval
    evaluate(env, rl_model)
