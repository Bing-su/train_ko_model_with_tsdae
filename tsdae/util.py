from sentence_transformers import SentenceTransformer, models


def build_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    :param model_name: str, model name
    :return: SentenceTransformer
    """
    transformer_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(), "cls"
    )
    model = SentenceTransformer(modules=[transformer_model, pooling_model])
    return model
