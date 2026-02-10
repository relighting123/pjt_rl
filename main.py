import argparse
import sys
import logging
from rts.models.train import run_training
from rts.models.inference import run_inference
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
    infer_parser.add_argument("--model_path", type=str, default="ppo_eqp_allocator")
    infer_parser.add_argument("--output", type=str, default="inference_results.csv")
    infer_parser.add_argument("--config", type=str, default="config.yaml")
    infer_parser.add_argument("--dry-run", action="store_true", help="Validate data and config without inference")

    args = parser.parse_args()

    # Initialize Logging and Config
    from rts.config.config_manager import load_config
    config = load_config(args.config if hasattr(args, 'config') else "config.yaml")
    
    setup_logging(log_dir=config.logging.log_dir)
    logger = logging.getLogger("rts.main")

    if not args.command:
        parser.print_help()
        return

    # Pre-flight check
    if not pre_flight_check(args.data_dir):
        logger.error("Pre-flight check failed. Exiting.")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run successful. Data and config are valid.")
        return

    if args.command == "train":
        run_training(args, config=config)
    elif args.command == "infer":
        run_inference(args, config=config)

if __name__ == "__main__":
    main()
