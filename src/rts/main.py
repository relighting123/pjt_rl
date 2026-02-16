import argparse
import sys
import logging
from rts.engine.train import run_training
from rts.engine.inference import run_inference
from rts.utils.logging_config import setup_logging
from rts.utils.system_checker import pre_flight_check

def main():
    parser = argparse.ArgumentParser(description="RTS: Reinforcement Training System for EQP Allocation Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train RL model")
    train_parser.add_argument("--data_dir", type=str, default="./data")
    train_parser.add_argument("--eval_data_dir", type=str, default=None)
    train_parser.add_argument("--scenario", type=str, default=None)
    train_parser.add_argument("--config", type=str, default="config.yaml")
    train_parser.add_argument("--dry-run", action="store_true", help="Validate data and config without training")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--mode", type=str, choices=["rl", "heuristic"], default="heuristic")
    infer_parser.add_argument("--data_dir", type=str, default="./data")
    infer_parser.add_argument("--scenario", type=str, default=None)
    infer_parser.add_argument("--timekey", type=str, default=None, help="OracleDB RULE_TIMEKEY")
    infer_parser.add_argument("--model_path", type=str, default="models/ppo_eqp_allocator")
    infer_parser.add_argument("--output", type=str, default="inference_results.csv")
    infer_parser.add_argument("--config", type=str, default="config.yaml")
    infer_parser.add_argument("--dry-run", action="store_true", help="Validate data and config without inference")

    # Sync-DB command
    sync_parser = subparsers.add_parser("sync-db", help="Download DB data to JSON for training")
    sync_parser.add_argument("--timekey", type=str, required=True, help="OracleDB RULE_TIMEKEY")
    sync_parser.add_argument("--data_dir", type=str, default="./data")
    sync_parser.add_argument("--config", type=str, default="config.yaml")

    # Serve command (API server)
    serve_parser = subparsers.add_parser("serve", help="Start API server with queue-based inference")
    serve_parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    # Initialize Logging and Config
    from rts.config.config_manager import load_config
    config = load_config(args.config if hasattr(args, 'config') else "config.yaml")
    
    setup_logging(log_dir=config.logging.log_dir)
    logger = logging.getLogger("rts.main")

    if not args.command:
        parser.print_help()
        return

    # serve command starts the API server (no pre-flight data check needed)
    if args.command == "serve":
        from rts.api.server import start_server
        start_server(config)
        return

    # Pre-flight check (for train/infer/sync-db that need data_dir)
    if not pre_flight_check(args.data_dir):
        logger.error("Pre-flight check failed. Exiting.")
        sys.exit(1)

    if args.command == "train":
        run_training(args, config=config)
    elif args.command == "infer":
        run_inference(args, config=config)
    elif args.command == "sync-db":
        from rts.database.db_manager import DBManager
        import json
        import os
        
        db = DBManager(config.db)
        logger.info(f"Downloading data for RULE_TIMEKEY: {args.timekey}")
        data = db.fetch_data(args.timekey)
        
        scn_name = f"scn_db_{args.timekey}"
        scn_path = os.path.join(args.data_dir, scn_name)
        os.makedirs(scn_path, exist_ok=True)
        
        # Save to JSONs
        with open(os.path.join(scn_path, 'equipment_capability.json'), 'w', encoding='utf-8') as f:
            json.dump({"capabilities": data['capabilities']}, f, indent=2)
        with open(os.path.join(scn_path, 'changeover_rules.json'), 'w', encoding='utf-8') as f:
            json.dump(data['changeover'], f, indent=2)
        with open(os.path.join(scn_path, 'equipment_inventory.json'), 'w', encoding='utf-8') as f:
            json.dump({"inventory": data['inventory']}, f, indent=2)
        with open(os.path.join(scn_path, 'plan_wip.json'), 'w', encoding='utf-8') as f:
            json.dump({"production": data['plan_wip']}, f, indent=2)
        if data['downtime']:
            with open(os.path.join(scn_path, 'equipment_downtime.json'), 'w', encoding='utf-8') as f:
                json.dump({"downtime": data['downtime']}, f, indent=2)
        
        logger.info(f"Successfully synced DB to {scn_path}")
        db.close()

if __name__ == "__main__":
    main()
