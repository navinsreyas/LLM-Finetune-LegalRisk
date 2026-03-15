"""
Unified training CLI for all three PEFT methods (QLoRA, DoRA, IA³).

Usage:
    python scripts/train.py --method qlora --config configs/qlora.yaml
    python scripts/train.py --method dora  --config configs/dora.yaml
    python scripts/train.py --method ia3   --config configs/ia3.yaml
    python scripts/train.py --method qlora --config configs/qlora.yaml --sanity-check
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.qlora_trainer import QLoRATrainer
from src.training.dora_trainer import DoRATrainer
from src.training.ia3_trainer import IA3Trainer


TRAINER_MAP = {
    "qlora": QLoRATrainer,
    "dora": DoRATrainer,
    "ia3": IA3Trainer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train LegalRisk-LLM with PEFT methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --method qlora --config configs/qlora.yaml
  python scripts/train.py --method qlora --config configs/qlora.yaml --sanity-check
  python scripts/train.py --method dora --config configs/dora.yaml
  python scripts/train.py --method ia3 --config configs/ia3.yaml
        """
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["qlora", "dora", "ia3"],
        required=True,
        help="PEFT method to use"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (configs/qlora.yaml, configs/dora.yaml, or configs/ia3.yaml)"
    )

    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run quick test with 10 examples and 1 epoch (~5 min)"
    )

    args = parser.parse_args()

    # Validate config file exists
    if not args.config.exists():
        print(f"[ERROR] Config file not found: {args.config}")
        print(f"[ERROR] Expected one of: configs/qlora.yaml, configs/dora.yaml, configs/ia3.yaml")
        sys.exit(1)

    if args.method not in str(args.config):
        print(f"[WARNING] Method '{args.method}' doesn't match config file '{args.config.name}'")
        print(f"[WARNING] This may cause unexpected results. Continue? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("[INFO] Exiting...")
            sys.exit(0)

    print("\n" + "="*70)
    print(f"  LegalRisk-LLM Phase 2A: {args.method.upper()} Training")
    if args.sanity_check:
        print(f"  [SANITY CHECK MODE]")
    print("="*70)
    print(f"  Config:  {args.config}")
    print(f"  Method:  {args.method}")
    print("="*70 + "\n")

    trainer_class = TRAINER_MAP[args.method]
    trainer = trainer_class(
        config_path=args.config,
        sanity_check=args.sanity_check
    )

    try:
        train_metrics, eval_metrics = trainer.run()

        print("\n" + "="*70)
        print(f"  SUCCESS: {args.method.upper()} training complete!")
        print("="*70)
        print(f"  Final train loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"  Final eval loss:  {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print("="*70 + "\n")

        if args.sanity_check:
            print("[INFO] Sanity check passed!")
            print("[INFO] You can now run full training by removing the --sanity-check flag")
        else:
            print(f"[INFO] Model adapter saved to: {trainer.config['output']['dir']}")
            print(f"[INFO] W&B logs: https://wandb.ai/{trainer.config.get('wandb', {}).get('entity', 'your-entity')}/{trainer.config.get('wandb', {}).get('project', 'legalrisk-llm')}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
