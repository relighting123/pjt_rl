import argparse
from stable_baselines3 import PPO
from env import TaktEnv
from expert import HeuristicExpert
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_production_results(logs, mode, scenario):
    plt.figure(figsize=(12, 6))
    
    # Aggregate produced_sum by timestamp and process to handle multiple products accurately
    # This prevents the "spiky" thorns caused by multiple rows at the same timestamp
    agg_logs = logs.groupby(['timestamp', 'process'])['produced_sum'].sum().reset_index()
    
    processes = agg_logs['process'].unique()
    for proc in processes:
        proc_data = agg_logs[agg_logs['process'] == proc].copy()
        proc_data = proc_data.sort_values('timestamp')
        plt.plot(proc_data['timestamp'], proc_data['produced_sum'], label=f'{proc} Total Cumulative', marker='o', markersize=3)
        
    plt.title(f"Production Progress ({mode.upper()}, Scenario: {scenario})")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Total Cumulative Production")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("logs", exist_ok=True)
    scenario_suffix = f"_{scenario}" if scenario else ""
    filename = f"logs/production_chart_{mode}{scenario_suffix}.png"
    plt.savefig(filename)
    print(f"Production chart saved to {filename}")
    plt.close()

def run_inference(mode="rl", data_dir="./data", model_path="ppo_takt_optimizer", scenario=None):
    print(f"--- Starting Inference (Mode: {mode}, Scenario: {scenario or 'Random'}) ---")
    
    env = TaktEnv(data_dir)
    
    # Load Policy
    if mode == "rl":
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Error: Model file {model_path}.zip not found. Please run train.py first.")
            return
        model = PPO.load(model_path)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        expert = HeuristicExpert(env)
        policy_fn = lambda obs: expert.select_action(obs)

    obs, _ = env.reset(options={"scenario": scenario} if scenario else None)
    # Get actual scenario if it was sampled randomly
    actual_scenario = env.current_scenario
    done = False
    
    print(f"{'Time':<6} | {'Product':<10} | {'Process':<10} | {'WIP':<6} | {'Prod':<6} | {'ActEqp':<6} | {'TarEqp':<6} | {'CO_Cnt':<6}")
    print("-" * 75)

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Print latest state
        history = env.get_logs()
        current_step = history['timestamp'].max()
        step_data = history[history['timestamp'] == current_step]
        
        for _, row in step_data.iterrows():
            print(f"{row['timestamp']:<6} | {row['product']:<10} | {row['process']:<10} | {row['wip']:<6.1f} | {row['production']:<6.1f} | {row['active_eqp']:<6.0f} | {row['target_eqp']:<6.0f} | {row['total_changeovers']:<6.0f}")
        
        done = terminated or truncated

    logs = env.get_logs()
    os.makedirs("logs", exist_ok=True)
    scenario_suffix = f"_{actual_scenario}" if actual_scenario else ""
    filename = f"logs/inference_results_{mode}{scenario_suffix}.csv"
    
    # Save the main logs
    logs.to_csv(filename, index=False)
    
    # Final Summary Calculation
    last_process = env.processes[-1]
    final_prods = logs[logs['timestamp'] == logs['timestamp'].max()]
    ach_df = final_prods[final_prods['process'] == last_process]
    ach_rate = ach_df['produced_sum'].sum() / (ach_df['plan'].sum() + 1e-6)
    
    total_work_minutes = 0
    for _, row in logs.iterrows():
        st = env.st_map.get((row['product'], row['process']), 0)
        total_work_minutes += row['production'] * st
    
    total_available_minutes = env.total_eqp * env.max_steps * 60
    utilization = total_work_minutes / total_available_minutes

    # Append Summary to CSV
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n")
        f.write(f"--- Inference Summary ({mode.upper()}) ---\n")
        f.write(f"Scenario,{actual_scenario}\n")
        f.write(f"Plan Achievement Rate (Last Step),{ach_rate:.2%}\n")
        f.write(f"Overall Equipment Utilization,{utilization:.2%}\n")
        f.write(f"Total Equipment Changeovers,{env.total_changeovers}\n")

    print(f"\n--- Inference Summary ({mode.upper()}) ---")
    print(f"Scenario: {actual_scenario}")
    print(f"Plan Achievement Rate (Last Step): {ach_rate:.2%}")
    print(f"Overall Equipment Utilization: {utilization:.2%}")
    print(f"Total Equipment Changeovers: {env.total_changeovers}")
    print(f"Detailed logs saved to {filename}")
    
    plot_production_results(logs, mode, actual_scenario)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using RL model or Heuristic algorithm.")
    parser.add_argument("--mode", type=str, choices=["rl", "heuristic"], default="rl", help="Selection of policy mode")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data files")
    parser.add_argument("--model", type=str, default="ppo_takt_optimizer", help="RL model path (without .zip)")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name (e.g., scn#1)")
    
    args = parser.parse_args()
    run_inference(mode=args.mode, data_dir=args.data_dir, model_path=args.model, scenario=args.scenario)
