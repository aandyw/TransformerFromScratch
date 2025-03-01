from pathlib import Path

import torch
import torch.nn as nn
from config import get_config, TransformerConfig
from dataset import BilingualDataset, create_causal_mask
from datasets import load_dataset
from model.transformer import Transformer, build_transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from typing import Callable, Iterator


class Trainer:
    """Transformer model training and validation"""

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config

        # Define the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Create model folder
        Path(self.config.model_folder).mkdir(parents=True, exist_ok=True)

        train_dl, val_dl, tokenizer_src, tokenizer_tgt = self._get_dataset()
        self.model = self._get_model(
            tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
        ).to(self.device)

        self.train_dl, self.val_dl = train_dl, val_dl
        self.tokenizer_src, self.tokenizer_tgt = tokenizer_src, tokenizer_tgt

        # 1. Ignore the padding ([PAD]) tokens
        # 2. Apply label smoothing - distributes X% of highest probability tokens to other tokens
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        ).to(self.device)

        self.writer = SummaryWriter(self.config.experiment_name)  # Tensorboard

        # Create adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, eps=1e-9
        )

        # Load existing model if it is specified / exists
        self._load_existing_model()

        self.global_step = 0
        self.initial_epoch = 0
        self.max_len = self.config.seq_len

    # Use word level tokenization
    def _get_all_sentences(self, dataset, lang) -> Iterator[str]:
        """Get all sentences in a given language from the dataset as a generator."""

        for item in dataset:
            yield item["translation"][lang]

    def _build_tokenizer(self, dataset, lang) -> Tokenizer:
        """Build / load a word-level tokenizer."""

        tokenizer_path = Path(self.config.tokenizer_file.format(lang))

        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            # Train tokenizer with special tokens:
            # Unknown, padding, start of sentence, end of sentence
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )

            tokenizer.train_from_iterator(
                self._get_all_sentences(dataset, lang), trainer=trainer
            )
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer

    def _get_dataset(self) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
        """Load the dataset to create dataloaders and tokenizers."""

        ds_raw = load_dataset(
            "opus_books",
            f"{self.config.lang_src}-{self.config.lang_tgt}",
            split="train",
        )

        # Build tokenizers
        tokenizer_src = self._build_tokenizer(ds_raw, self.config.lang_src)
        tokenizer_tgt = self._build_tokenizer(ds_raw, self.config.lang_tgt)

        # Keep 90% for training and 10% for validation
        train_size = int(0.9 * len(ds_raw))
        val_size = len(ds_raw) - train_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

        # Create datasets
        train_dataset = BilingualDataset(
            train_ds_raw,
            tokenizer_src,
            tokenizer_tgt,
            self.config.lang_src,
            self.config.lang_tgt,
            self.config.seq_len,
        )
        val_dataset = BilingualDataset(
            val_ds_raw,
            tokenizer_src,
            tokenizer_tgt,
            self.config.lang_src,
            self.config.lang_tgt,
            self.config.seq_len,
        )

        # Find max sentence in src and tgt languages
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = tokenizer_src.encode(
                item["translation"][self.config.lang_src]
            ).ids
            tgt_ids = tokenizer_src.encode(
                item["translation"][self.config.lang_tgt]
            ).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of source sentence: {max_len_tgt}")

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

    def _get_model(self, vocab_src_len, vocab_tgt_len) -> Transformer:
        model = build_transformer(
            vocab_src_len,
            vocab_tgt_len,
            self.config.seq_len,
            self.config.seq_len,
            self.config.d_model,
        )
        return model

    def _load_existing_model(self) -> None:
        """Load an existing model from a given epoch."""

        if self.config.load_from:
            epoch_name = self.config.load_from
            model_filename = self.config.get_weights_file_path(epoch_name)
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)

            self.initial_epoch = state["epoch"] + 1
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

    ### TRAINING CODE ###
    def train(self) -> None:
        """Train the transformer model."""

        for epoch in range(self.initial_epoch, self.config.num_epochs):
            batch_iterator = tqdm(self.train_dl, desc=f"Processing epoch {epoch:02d}")
            for batch in batch_iterator:
                self.model.train()

                encoder_input = batch["encoder_input"].to(self.device)  # (bs, seq_len)
                decoder_input = batch["decoder_input"].to(self.device)  # (bs, seq_len)

                # Attention mask (hide padding tokens)
                encoder_mask = batch["encoder_mask"].to(
                    self.device
                )  # (bs, 1, 1, seq_len)

                # Casual mask (hide padding tokens and future)
                decoder_mask = batch["decoder_mask"].to(
                    self.device
                )  # (bs, 1, seq_len, seq_len)

                # Input passthrough
                encoder_output = self.model.encode(
                    encoder_input, encoder_mask
                )  # (bs, seq_len, d_model)
                decoder_output = self.model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (bs, seq_len, d_model)
                proj_output = self.model.project(
                    decoder_output
                )  # (bs, seq_len, tgt_vocab_size)

                label = batch["label"].to(self.device)  # (bs, seq_len)

                # (bs, seq_len, tgt_vocab_size) -> (bs * seq_len, tgt_vocab_size)
                pred = proj_output.view(-1, self.tokenizer_tgt.get_vocab_size())
                gt = label.view(-1)  # Ground truth
                loss = self.loss_fn(pred, gt)
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss in tensorboard
                self.writer.add_scalar("train loss", loss.item(), self.global_step)
                self.writer.flush()

                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            model_filename = self.config.get_weights_file_path(f"{epoch:02d}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                },
                model_filename,
            )

            self.validate(
                lambda msg: batch_iterator.write(msg), self.global_step, self.writer
            )

    ### VALIDATION CODE ###
    def greedy_decode(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ):
        """
        Greedy decode for efficient validation.
        Highest probability token is selected at each step as the next word.
        """

        sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")

        # Precompute the encoder output and reuse it for every token we get from decoder
        encoder_output = self.model.encode(source, source_mask)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(self.device)

        # Keep predicting until we reach EOS or max_len
        while True:
            if decoder_input.size(1) == self.config.seq_len:
                break

            # Build the mask
            decoder_mask = create_causal_mask(decoder_input.size(1))
            decoder_mask = decoder_mask.type_as(source_mask).to(self.device)

            # Calculate the output of the decoder
            out = self.model.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )

            # Get the next token
            prob = self.model.project(out[:, -1])  # the project of the last token

            # Select the token with the highest probability (because it is greedy search)
            _, next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(self.device),
                ],
                dim=1,
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)  # remove batch dimension

    def validate(
        self, print_msg: Callable, global_step: int, writer: SummaryWriter
    ) -> None:
        """Run transformer model on the validation dataset."""

        self.model.eval()  # Put in eval mode
        count = 0

        # source_texts = []
        # expected = []
        # predicted = []

        CONSOLE_WIDTH = 80

        with torch.no_grad():
            for batch in self.val_dl:
                count += 1

                # (bs, seq_len)
                encoder_input = batch["encoder_input"].to(self.device)

                # (bs, 1, 1, seq_len)
                encoder_mask = batch["encoder_mask"].to(self.device)

                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                # Get generation
                model_out = self.greedy_decode(encoder_input, encoder_mask)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]

                # Detach model_out from computational graph
                model_out_tensor = model_out.detach().cpu()
                model_out_array = model_out_tensor.numpy()
                model_out_text = self.tokenizer_tgt.decode(model_out_array)

                # source_texts.append(source_text)
                # expected.append(target_text)
                # predicted.append(model_out_text)

                # Print to the console
                print_msg("-" * CONSOLE_WIDTH)
                print_msg(f"SOURCE: {source_text}")
                print_msg(f"TARGET: {target_text}")
                print_msg(f"PREDICTED: {model_out_text}")


if __name__ == "__main__":
    config = get_config()
    trainer = Trainer(config)
    trainer.train()
