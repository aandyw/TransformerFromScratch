from typing import Any, Dict

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset, Subset


def create_causal_mask(size: int) -> Tensor:
    """
    Causal mask used only in decoder to ensure that future is masked.
    https://discuss.huggingface.co/t/difference-between-attention-mask-and-causal-mask/104922
    """

    # Diagonal=1 to get a mask that does not include the main diagonal and only the
    # upper triangular part of the matrix excluding the main diagonal
    ones_matrix = torch.ones(1, size, size)
    mask = torch.triu(ones_matrix, diagonal=1).type(torch.int)

    # The above returns the mask for the upper diagonal which we DON'T want to include
    # in the causal mask. We want it False. Therefore, we use `mask == 0`.
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset: Subset,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        lang_src: str,
        lang_tgt: str,
        seq_len: int,
    ):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        # Vocab can be longer than 32 bits so we use int64
        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        # Get the source and target text pair
        src_tgt_pair = self.dataset[index]

        # Extract the individual source and target text
        src_text = src_tgt_pair["translation"][self.lang_src]
        tgt_text = src_tgt_pair["translation"][self.lang_tgt]

        # Convert tokens to ids
        enc_input = self.tokenizer_src.encode(src_text).ids
        dec_input = self.tokenizer_tgt.encode(tgt_text).ids

        # -2 to account for the start and end tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input) - 2

        # -1 to account for the end token
        dec_num_padding_tokens = self.seq_len - len(dec_input) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long.")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(enc_num_padding_tokens),  # Pad to reach seq_len
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input, dtype=torch.int64),
                self.pad_token.repeat(dec_num_padding_tokens),  # Pad to reach seq_len
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(dec_num_padding_tokens),  # Pad to reach seq_len
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        # Causal mask to prevent attending to future
        casual_mask = create_causal_mask(decoder_input.size(0))
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = decoder_mask & casual_mask

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,  # (1, 1, seq_len)
            "decoder_mask": decoder_mask,  # (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
