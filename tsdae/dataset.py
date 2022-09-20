from typing import TYPE_CHECKING, Callable, Optional

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

    :param dataset: huggingface datasets 라이브러리의 Dataset 객체
    :param text_col: str, dataset에서 사용할 텍스트 행 이름
    :param p: float, 기본 noise 함수에서 적용될 확률값. 0 <= p <= 1, default: 0.6
    :param noise_fn: Callable[[str], str], 텍스트에 적용할 noise 함수. default: None.
        None일 경우, 기본 noise 함수를 사용합니다.
    """

    def __init__(
        self,
        dataset: Dataset,
        text_col: str,
        p: float = 0.6,
        noise_fn: Optional[Callable[[str], str]] = None,
    ):
        self.dataset = dataset
        self.text_col = text_col
        self.p = p
        assert 0 <= p <= 1
        if noise_fn is None:
            self.noise_fn = self.delete
        else:
            self.noise_fn = noise_fn

        self.kiwi = Kiwi(model_type="sbg")

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
