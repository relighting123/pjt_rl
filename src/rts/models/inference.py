import argparse
import logging
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from ..env.factory_env import ProductionEnv
from .expert import HeuristicExpert
from ..config.config_manager import Config
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

def run_inference(args, config: Config = None):
    config = config or Config()
    
    # 1. Load Data (DB or File)
    raw_data = None
    rule_timekey = getattr(args, 'timekey', None)
    
    if config.db.enabled and rule_timekey:
        from ..data.db_manager import DBManager
        db = DBManager(config.db)
        logger.info(f"Fetching data from DB for RULE_TIMEKEY: {rule_timekey}")
        raw_data = db.fetch_data(rule_timekey)
    
    logger.info(f"Initializing Inference Environment for Scenario: {args.scenario or rule_timekey or 'Random'}")
    env = ProductionEnv(args.data_dir, fixed_scenario=args.scenario, config=config.env, raw_data=raw_data)
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
    
    # 2. Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("logs", f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Save Results (CSV and DB)
    csv_path = os.path.join(output_dir, "result.csv")
    try:
        logs.to_csv(csv_path, index=False)
        logger.info(f"Inference logs saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV logs: {e}")

    if config.db.enabled and rule_timekey:
        try:
            db.upload_results(rule_timekey, logs.to_dict('records'))
            db.close()
        except Exception as e:
            logger.error(f"Failed to upload results to DB: {e}")
            
    # 4. Generate Production Chart
    try:
        plt.figure(figsize=(12, 7))
        
        scenario_products = env.products[:env.real_num_prods]
        scenario_processes = env.processes[:env.real_num_procs]
        
        # Use a colormap that can handle more entries
        cm = plt.get_cmap('tab20')
        line_idx = 0
        
        for p_idx, prod in enumerate(scenario_products):
            # Get processes for this specific product from logs
            prod_data = logs[logs['product'] == prod]
            prod_procs = prod_data['process'].unique()
            
            # Sort processes based on their order in scenario_processes
            sorted_procs = [s for s in scenario_processes if s in prod_procs]
            if not sorted_procs: continue
            
            for s_idx, proc in enumerate(sorted_procs):
                proc_logs = prod_data[prod_data['process'] == proc]
                if proc_logs.empty: continue
                
                color = cm(line_idx % 20)
                line_idx += 1
                
                is_last = (proc == sorted_procs[-1])
                label = f"{prod}-{proc}"
                
                alpha = 1.0
                linestyle = '-'
                linewidth = 2 if is_last else 1.5
                
                plt.plot(proc_logs['timestamp'], proc_logs['produced_sum'], 
                         label=label, alpha=alpha, linestyle=linestyle, linewidth=linewidth, color=color)
                
                # Add plan goal line for the LAST process of each product
                if is_last:
                    plan_val = float(proc_logs['plan'].iloc[0])
                    plt.axhline(y=plan_val, color=color, linestyle=':', alpha=0.6)
                    # Label the goal
                    plt.text(0, plan_val + 0.5, f"{prod} Goal ({int(plan_val)})", color=color, fontsize=8, verticalalignment='bottom', fontweight='bold')
        
        plt.title(f"Cumulative Production Flow (Scenario: {args.scenario or rule_timekey or 'Random'})")
        plt.xlabel("Step (Hour)")
        plt.ylabel("Cumulative Produced Qty")
        
        # Set x-axis ticks to 1-unit intervals
        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        chart_path = os.path.join(output_dir, "cumulative_production.png")
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Cumulative production chart saved to {chart_path}")
    except Exception as e:
        logger.error(f"Failed to generate production chart: {e}")
        
    # Final Metrics
    final_prods = logs[logs['timestamp'] == logs['timestamp'].max()]
    
    # Use the actual last process of the scenario
    current_scenario_procs = getattr(env.loader, 'get_processes', lambda: [])()
    if current_scenario_procs:
        last_process = current_scenario_procs[-1]
    else:
        last_process = env.processes[env.real_num_procs - 1]
        
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
