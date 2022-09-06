from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from datasets import Dataset
from kiwipiepy import Kiwi
from sentence_transformers import InputExample
from torch.utils import data

if TYPE_CHECKING:
    from kiwipiepy import Token


class KoDenoisingAutoEncoderDataset(data.Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format:
        texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss:
        Here, a decoder tries to re-construct the sentence without noise.

    :param dataset: dataset from datasets library
    :param text_col: str, text column name of the dataset
    :param p: float, probability of applying noise_fn, 0 <= p <= 1, default: 0.6
    """

    def __init__(
        self,
        dataset: Dataset,
        text_col: str,
        p: float = 0.6,
        noise_fn: Callable[[str], str] | None = None,
    ):
        self.dataset = dataset
        self.text_col = text_col
        self.p = p
        assert 0 <= p <= 1
        if noise_fn is None:
            self.noise_fn = self.delete
        else:
            self.noise_fn = noise_fn

        self.kiwi = Kiwi(num_workers=1, model_type="sbg")

    def __getitem__(self, idx: int) -> InputExample:
        text: str = self.dataset[idx][self.text_col]
        return InputExample(texts=[self.noise_fn(text), text])

    def __len__(self):
        return len(self.dataset)

    # Deletion noise.
    def delete(self, text: str, /) -> str:
        tokens: list[Token] = self.kiwi.tokenize(text)
        n = len(tokens)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > self.p
        if sum(keep_or_not) == 0:
            keep_or_not[
                np.random.choice(n)
            ] = True  # guarantee that at least one word remains
        tokens_processed = [tokens[i] for i in range(n) if keep_or_not[i]]
        words_processed = self.kiwi.join(tokens_processed, lm_search=False)
        return words_processed
