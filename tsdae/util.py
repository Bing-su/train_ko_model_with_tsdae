import os
from platform import uname
from typing import Optional

import torch
from pytorch_optimizer import load_optimizer
from sentence_transformers import SentenceTransformer, models
from torch.optim import SGD, Adam, AdamW


def build_sentence_transformer(
    model_name: str, max_seq_length: Optional[int] = None
) -> SentenceTransformer:
    """
    :param model_name: str, huggingface 모델 이름
    :param max_seq_length: int, 모델이 입력으로 받을 수 있는 최대 길이
    ```
    CUDA error: device-side assert triggered
    CUDA kernel errors might be asynchronously reported at some other API call,
    so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
    ```
    위 에러 발생시 수동으로 줄여보십시오.

    :return: SentenceTransformer
    """
    transformer_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(), "cls"
    )
    model = SentenceTransformer(modules=[transformer_model, pooling_model])
    return model


def is_in_wsl() -> bool:
    """
    :return: bool, 현재 환경이 WSL인지 여부
    """
    return "microsoft-standard" in uname().release


def create_optimizer(name: str):
    name = name.lower()

    if name == "adam":
        return Adam
    elif name == "adamw":
        return AdamW
    elif name == "sgd":
        return SGD
    elif name in ("adam_bnb", "adamw_bnb"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BNB optimizers")

        if is_in_wsl():
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"

        try:
            from bitsandbytes.optim import Adam8bit, AdamW8bit

            if name == "adam_bnb":
                return Adam8bit
            else:
                return AdamW8bit

        except ImportError as e:
            raise ImportError("bitsandbytes를 먼저 설치해주세요") from e
    else:
        return load_optimizer(name)
