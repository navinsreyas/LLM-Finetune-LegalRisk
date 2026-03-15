"""QLoRA Trainer: LoRA + 4-bit NF4 quantization baseline."""

from peft import LoraConfig, TaskType
from src.training.base_trainer import BasePEFTTrainer


class QLoRATrainer(BasePEFTTrainer):
    """QLoRA: r=16, alpha=32, 4-bit NF4, targets all attention + MLP layers (~2.6M trainable params)."""

    def get_peft_config(self):
        peft_config = self.config["peft"]

        return LoraConfig(
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            bias=peft_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_config["target_modules"],
            use_dora=False,
            use_rslora=peft_config.get("use_rslora", False),
        )
