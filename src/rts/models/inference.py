import argparse
import logging
import pandas as pd
from ..env.factory_env import TaktEnv
from .expert import HeuristicExpert
from ..config.config_manager import Config
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

def run_inference(args, config: Config = None):
    config = config or Config()
    
    logger.info(f"Initializing Inference Environment for Scenario: {args.scenario or 'Random'}")
    env = TaktEnv(args.data_dir, fixed_scenario=args.scenario, config=config.env)
    obs, _ = env.reset()
    
    if args.mode == "rl":
        logger.info(f"Loading RL Model: {args.model_path}")
        model = PPO.load(args.model_path)
    else:
        logger.info("Using Heuristic Expert Mode")
        model = HeuristicExpert(env)

    done = False
    while not done:
        if args.mode == "rl":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = model.select_action(obs)
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    logs = env.get_logs()
    try:
        logs.to_csv(args.output, index=False)
        logger.info(f"Inference logs saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save logs: {e}")
        
    # Final Metrics
    final_prods = logs[logs['timestamp'] == logs['timestamp'].max()]
    last_process = env.processes[-1]
    ach_df = final_prods[final_prods['process'] == last_process]
    ach_rate = ach_df['produced_sum'].sum() / (ach_df['plan'].sum() + 1e-6)
    
    total_work_minutes = 0
    for _, row in logs.iterrows():
        st = env.st_map.get((row['product'], row['process']), 0)
        total_work_minutes += row['production'] * st
    total_available_minutes = env.total_eqp * env.max_steps * 60
    utilization = total_work_minutes / total_available_minutes
    
    logger.info(f"\n--- Inference Results ---")
    logger.info(f"Plan Achievement Rate: {ach_rate:.2%}")
    logger.info(f"Overall Equipment Utilization: {utilization:.2%}")
    logger.info(f"Total Equipment Changeovers: {env.total_changeovers}\n")
    
    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference for EQP Allocation Optimizer.")
    parser.add_argument("--mode", type=str, choices=["rl", "heuristic"], default="heuristic")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="ppo_eqp_allocator")
    parser.add_argument("--output", type=str, default="inference_results.csv")

    args = parser.parse_args()
    run_inference(args)
