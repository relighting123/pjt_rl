import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm

from ..env.factory_env import TaktEnv
from .expert import HeuristicExpert
from ..data.data_loader import DataLoader
from ..config.config_manager import Config
from ..utils.seed_manager import set_seed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

logger = logging.getLogger(__name__)

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            from ..utils.system_checker import log_system_status
            log_system_status()
            
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append(info['episode']['r'])
        return True

def plot_convergence(rewards, filename="convergence_chart.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    if len(rewards) > 10:
        yield_avg = pd.Series(rewards).rolling(window=10).mean()
        plt.plot(yield_avg, label='Rolling Average (10)', color='red')
    plt.title("RL Training Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    logger.info(f"Convergence chart saved to {filename}")
    plt.close()

def train_rl(env, bc_model=None, model_name: str = "ppo_eqp_allocator", config: Config = None):
    train_cfg = config.train if config else None
    
    logger.info(f"--- Starting RL Fine-tuning (PPO) for model '{model_name}' ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=train_cfg.learning_rate if train_cfg else 3e-4,
        n_steps=train_cfg.n_steps if train_cfg else 1024,
        batch_size=train_cfg.batch_size if train_cfg else 128,
        ent_coef=train_cfg.ent_coef if train_cfg else 0.01,
        seed=train_cfg.seed if train_cfg else None,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    )

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=train_cfg.timesteps if train_cfg else 15000, callback=callback)
    model.save(model_name)

    if callback.rewards:
        plot_convergence(callback.rewards, filename=f"convergence_chart_{model_name}.png")

    return model

def make_env(data_dir, scenario, config):
    def _init():
        return TaktEnv(data_dir, fixed_scenario=scenario, config=config)
    return _init

def run_training(args, config: Config = None):
    config = config or Config()
    
    # 0. Set Seed
    set_seed(config.train.seed)

    logger.info(f"Initializing Multi-Scenario Environment from {args.data_dir}")
    num_envs = os.cpu_count() or 1
    # For Windows, SubprocVecEnv might need if __name__ == '__main__': 
    # but we are inside run_training which is called from main.
    if num_envs > 1 and os.name != 'nt': # SubprocVecEnv can be unstable on Windows in some setups
        env = SubprocVecEnv([make_env(args.data_dir, args.scenario, config.env) for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(args.data_dir, args.scenario, config.env)])
    
    # single env for expert data generation
    single_env = TaktEnv(args.data_dir, fixed_scenario=args.scenario, config=config.env)
    
    # 1. Behavior Cloning (BC) Pre-training
    expert = HeuristicExpert(single_env)
    logger.info("--- Starting Behavior Cloning (BC) with Multi-Scenario Expert ---")
    
    def generate_multi_scenario_trajectories(expert, num_episodes=200):
        data = []
        scenarios = expert.env.scenarios
        for i in tqdm(range(num_episodes), desc="Generating expert trajectories"):
            scn = scenarios[i % len(scenarios)]
            obs, _ = expert.env.reset(options={"scenario": scn})
            done = False
            while not done:
                action = expert.select_action(obs)
                next_obs, reward, terminated, truncated, _ = expert.env.step(action)
                data.append((obs, action))
                obs = next_obs
                done = terminated or truncated
        return data

    trajectories = generate_multi_scenario_trajectories(expert, num_episodes=200)
    obs_data, action_data = zip(*trajectories)
    obs_tensor = torch.tensor(np.array(obs_data), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_data), dtype=torch.long)
    dataset = TensorDataset(obs_tensor, action_tensor)
    loader = TorchDataLoader(dataset, batch_size=64, shuffle=True)
    
    model_bc = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )
    optimizer = optim.Adam(model_bc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(30):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            output = model_bc(batch_obs)
            loss = criterion(output, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            logger.info(f"BC Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model_bc.state_dict(), "bc_model.pth")

    # 2. RL Fine-tuning
    model_name = "ppo_eqp_allocator"
    rl_model = train_rl(env, bc_model=model_bc, model_name=model_name, config=config)
    
    # 3. Evaluation
    eval_dir = args.eval_data_dir if args.eval_data_dir else args.data_dir
    logger.info(f"\n=== Final Evaluation on All Scenarios from {eval_dir} ===")
    
    scenarios = DataLoader.list_scenarios(eval_dir)
    eval_env = TaktEnv(eval_dir, fixed_scenario=None)
    
    results = []
    for scn in scenarios:
        logger.info(f"\n[Scenario: {scn}]")
        obs, _ = eval_env.reset(options={"scenario": scn})
        done = False
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        logs = eval_env.get_logs()
        final_prods = logs[logs['timestamp'] == logs['timestamp'].max()]
        last_process = eval_env.processes[-1]
        ach_df = final_prods[final_prods['process'] == last_process]
        ach_rate = ach_df['produced_sum'].sum() / (ach_df['plan'].sum() + 1e-6)
        
        total_work_minutes = 0
        for _, row in logs.iterrows():
            st = eval_env.st_map.get((row['product'], row['process']), 0)
            total_work_minutes += row['production'] * st
        total_available_minutes = eval_env.total_eqp * eval_env.max_steps * 60
        utilization = total_work_minutes / total_available_minutes
        
        logger.info(f"Plan Achievement Rate: {ach_rate:.2%}")
        logger.info(f"Equipment Utilization: {utilization:.2%}")
        logger.info(f"Total Changeovers: {eval_env.total_changeovers}")
        
        results.append({
            "scenario": scn,
            "achievement": ach_rate,
            "utilization": utilization,
            "changeovers": eval_env.total_changeovers
        })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL EQP Allocation Optimizer.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for training data files")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="Directory for evaluation data files")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario to train on")
    parser.add_argument("--timesteps", type=int, default=60000, help="Total timesteps for RL training")

    args = parser.parse_args()
    run_training(args)
