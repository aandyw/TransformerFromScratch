from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Transformer training configuration
    """

    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 1e-4
    seq_len: int = 350  # Max sequence length
    d_model: int = 512
    lang_src: str = "en"  # Source language: English
    lang_tgt: str = "it"  # Target language: Italian
    model_folder: str = "weights"
    model_filename: str = "transformer_"  # Base filename for saved weights
    load_from: Optional[str] = None  # Load from this epoch (e.g.)
    tokenizer_file: str = "tokenizer_0.json"
    experiment_name: str = "runs/transformer"

    def get_weights_file_path(self, epoch_name: str) -> str:
        """
        Get the file path for the model weights corresponding to a given epoch.

        Parameters:
            epoch_name (str): The epoch name to get weights file path for.

        Returns:
            str: The complete file path for the weights file.
        """
        return str(
            Path(".") / self.model_folder / f"{self.model_filename}{epoch_name}.pt"
        )


def get_config() -> TransformerConfig:
    """
    Retrieve the default configuration for the Transformer model training.

    Returns:
        TransformerConfig: An instance of TransformerConfig with default values.
    """
    return TransformerConfig()


if __name__ == "__main__":
    # Test get config
    config = get_config()
    epoch = 1
    weights_path = config.get_weights_file_path(epoch)
    print(f"Weights file path for epoch {epoch}: {weights_path}")
