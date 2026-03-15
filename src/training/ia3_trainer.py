"""IA³ Trainer: scales activations via learned vectors instead of weight matrices (~100x fewer params than LoRA)."""

from peft import IA3Config, TaskType
from src.training.base_trainer import BasePEFTTrainer


class IA3Trainer(BasePEFTTrainer):
    """
    IA³: learns scaling vectors for k_proj, v_proj, down_proj instead of low-rank weight matrices.

    Trade-offs vs QLoRA/DoRA:
    - Fewer trainable params (~0.03M vs ~2.6M) — less expressive
    - No quantization (full fp16) — uses more VRAM
    - Higher learning rate needed (3e-3 vs 2e-4)
    """

    def get_peft_config(self):
        peft_config = self.config["peft"]

        return IA3Config(
            target_modules=peft_config["target_modules"],
            feedforward_modules=peft_config["feedforward_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
