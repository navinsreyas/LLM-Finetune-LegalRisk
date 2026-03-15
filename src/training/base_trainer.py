"""
Base PEFT Trainer shared by QLoRA, DoRA, and IA³. Subclasses only override get_peft_config().

Notes:
- Llama-3.2 has no pad_token by default; we set it to eos_token
- RTX 4060 lacks bfloat16 support; use fp16 only
- packing=False because legal clauses vary widely in length (50-500 tokens)
"""

import yaml
import torch
import wandb
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from src.training.data_formatter import load_and_format_dataset


class BasePEFTTrainer:
    """
    Base class for all PEFT training methods.

    Subclasses must implement:
        - get_peft_config(): Return LoraConfig or IA3Config

    Usage:
        trainer = QLoRATrainer("configs/qlora.yaml")
        trainer.train()  # Does everything: load, train, evaluate, save
    """

    def __init__(self, config_path: str | Path, sanity_check: bool = False):
        """
        Initialize trainer with config file.

        Args:
            config_path: Path to YAML config (qlora.yaml, dora.yaml, or ia3.yaml)
            sanity_check: If True, train on only 10 examples for rapid testing
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sanity_check = sanity_check
        self.method_name = self.config.get("method", "unknown")

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None

        print(f"\n{'='*70}")
        print(f"  {self.method_name.upper()} Training - LegalRisk-LLM Phase 2A")
        if self.sanity_check:
            print(f"  [SANITY CHECK MODE: 10 examples, 1 epoch]")
        print(f"{'='*70}\n")

    def get_peft_config(self):
        """
        Define PEFT configuration (LoRA or IA³).

        MUST be overridden in subclass.

        Returns:
            peft.LoraConfig or peft.IA3Config
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_peft_config()")

    def load_model_and_tokenizer(self):
        """
        Load base model with optional 4-bit quantization and fix tokenizer.

        Memory optimization strategy:
        - 4-bit NF4 quantization: 3B params × 0.5 bytes = 1.5GB (vs 6GB fp16)
        - Double quantization: Reduces to ~1.3GB
        - Gradient checkpointing: Saves 40% activation memory
        - Total peak VRAM: ~4.5GB (safe for 8GB)
        """
        model_config = self.config["model"]
        model_name = model_config["name"]

        print(f"[MODEL] Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # CRITICAL FIX: Llama-3.2 has no pad_token by default
        # Without this, batch padding crashes during training
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"[MODEL] Fixed pad_token (set to eos_token)")

        if model_config.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.float16,  # RTX 4060 lacks bf16
                bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            )
            print(f"[MODEL] Using 4-bit NF4 quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation=model_config.get("attn_implementation", "sdpa"),
                trust_remote_code=model_config.get("trust_remote_code", False),
            )
        else:
            print(f"[MODEL] Using full fp16 (no quantization)")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation=model_config.get("attn_implementation", "sdpa"),
                trust_remote_code=model_config.get("trust_remote_code", False),
            )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # CRITICAL FIX: Convert any BFloat16 parameters to Float16
        # RTX 4060 doesn't fully support BFloat16 operations
        for param in self.model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

        print(f"[MODEL] Loaded successfully")
        print(f"[MODEL] Total parameters: {self.model.num_parameters():,}")

    def prepare_model_for_training(self):
        """
        Prepare model for k-bit training and apply PEFT adapter.

        Critical steps:
        1. prepare_model_for_kbit_training (if quantized)
        2. Disable cache (required for gradient checkpointing)
        3. Apply PEFT config
        4. Print trainable parameter count
        """
        # Prepare for k-bit training if quantized
        if self.config["model"].get("load_in_4bit", False):
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True
            )
            print(f"[PEFT] Model prepared for k-bit training")
        else:
            self.model.gradient_checkpointing_enable()
            print(f"[PEFT] Gradient checkpointing enabled")

        self.model.config.use_cache = False

        peft_config = self.get_peft_config()

        self.model = get_peft_model(self.model, peft_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        print(f"[PEFT] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.4f}%)")

    def load_datasets(self):
        """Load and format train/val datasets using data_formatter."""
        data_config = self.config["data"]

        limit = 10 if self.sanity_check else None

        self.train_dataset = load_and_format_dataset(data_config["train_path"], limit=limit)
        self.val_dataset = load_and_format_dataset(data_config["val_path"], limit=limit)

        print(f"[DATA] Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train(self):
        """
        Execute the training loop using SFTTrainer.

        SFTTrainer handles:
        - Chat template application (apply_chat_template)
        - Tokenization with proper special tokens
        - Batch collation
        - Training loop
        - Evaluation
        - Checkpointing
        """
        train_config = self.config["training"]
        output_dir = Path(self.config["output"]["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.sanity_check:
            train_config = train_config.copy()  # Don't modify original
            train_config["num_train_epochs"] = 1
            train_config["save_steps"] = 5
            train_config["eval_steps"] = 5

        wandb_config = self.config.get("wandb", {})
        run_name = f"{self.method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.sanity_check:
            run_name += "_sanity"

        wandb.init(
            project=wandb_config.get("project", "legalrisk-llm"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            config=self.config,
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=train_config["num_train_epochs"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", train_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=float(train_config["learning_rate"]),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=train_config.get("warmup_ratio", 0.03),
            weight_decay=train_config.get("weight_decay", 0.01),
            max_grad_norm=train_config.get("max_grad_norm", 1.0),

            # Memory optimization
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            fp16=train_config.get("fp16", True),
            bf16=train_config.get("bf16", False),
            optim=train_config.get("optim", "paged_adamw_8bit"),

            # Logging and evaluation
            logging_steps=train_config.get("logging_steps", 10),
            logging_first_step=train_config.get("logging_first_step", True),
            eval_strategy=train_config.get("eval_strategy", "steps"),
            eval_steps=train_config.get("eval_steps", 50),

            # Checkpointing
            save_strategy=train_config.get("save_strategy", "steps"),
            save_steps=train_config.get("save_steps", 100),
            save_total_limit=train_config.get("save_total_limit", 3),
            load_best_model_at_end=train_config.get("load_best_model_at_end", True),
            metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=train_config.get("greater_is_better", False),

            # Data handling
            group_by_length=train_config.get("group_by_length", True),
            dataloader_pin_memory=True,

            # Reproducibility
            seed=42,
            data_seed=42,

            # Reporting
            report_to="wandb",
            run_name=run_name,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
        )

        print(f"\n[TRAIN] Starting training...")
        print(f"[TRAIN] Epochs: {train_config['num_train_epochs']}")
        print(f"[TRAIN] Batch size: {train_config['per_device_train_batch_size']}")
        print(f"[TRAIN] Gradient accumulation: {train_config['gradient_accumulation_steps']}")
        print(f"[TRAIN] Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
        print(f"[TRAIN] Learning rate: {train_config['learning_rate']}")
        print()

        train_result = self.trainer.train()

        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        print(f"\n[TRAIN] Training complete!")
        print(f"[TRAIN] Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")

        return metrics

    def evaluate(self):
        """Run final evaluation on validation set."""
        print(f"\n[EVAL] Running final evaluation...")
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        print(f"[EVAL] Final val loss: {metrics['eval_loss']:.4f}")
        return metrics

    def save(self):
        """Save adapter weights and training config."""
        output_dir = Path(self.config["output"]["dir"])

        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        config_save_path = output_dir / "training_config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        adapter_size_mb = sum(
            f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
        ) / 1024 / 1024

        print(f"\n[SAVE] Model saved to {output_dir}")
        print(f"[SAVE] Adapter size: {adapter_size_mb:.1f} MB")

    def run(self):
        """Execute the complete training pipeline."""
        try:
            self.load_model_and_tokenizer()
            self.prepare_model_for_training()
            self.load_datasets()

            train_metrics = self.train()
            eval_metrics = self.evaluate()
            self.save()

            wandb.finish()

            print(f"\n{'='*70}")
            print(f"  {self.method_name.upper()} Training Complete!")
            print(f"{'='*70}")
            print(f"  Final train loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
            print(f"  Final val loss:   {eval_metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"  Model saved to:   {self.config['output']['dir']}")
            print(f"{'='*70}\n")

            return train_metrics, eval_metrics

        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            wandb.finish(exit_code=1)
            raise
