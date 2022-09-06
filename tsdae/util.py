from typing import Optional

from sentence_transformers import SentenceTransformer, models


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
