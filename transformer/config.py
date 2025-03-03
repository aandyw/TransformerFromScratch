from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class TransformerConfig:
    """
    Transformer training configuration
    """

    batch_size: int = 16
    num_epochs: int = 10
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


def get_config(overrides: Optional[Dict[str, Any]] = None) -> TransformerConfig:
    """
    Retrieve the default configuration for the Transformer model training.
    Optionally override configuration values by passing in a dictionary.

    Parameters:
        overrides (Optional[Dict[str, Any]]): Dictionary with keys corresponding to the
            TransformerConfig fields that should be overridden.

    Returns:
        TransformerConfig: An instance of TransformerConfig with default values.
    """
    config = TransformerConfig()
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise KeyError(f"Invalid config key: {key}")
    return config


if __name__ == "__main__":
    # Test get config
    config = get_config({"batch_size": 32, "num_epochs": 5})
    epoch = "epoch_1"
    weights_path = config.get_weights_file_path(epoch)
    print(f"Weights file path for epoch {epoch}: {weights_path}")
