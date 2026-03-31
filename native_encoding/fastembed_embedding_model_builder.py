"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/native_encoding/fastembed_embedding_model_builder.py

References:
    - https://qdrant.github.io/fastembed/
    - https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models

Synopsis:
    基于 FastEmbed 的 embedding model builder 。

Notes:
    直接使用 fastembed 的 embedding model 。

    该方法与 qdrant 高度集成。

    注意:
        - onnx: 运行模型需要有 ONNX Runtime 。
        - supported: 简单方法需要官方支持，检查官方文档。自定义模型以后实现构建方法。
"""

from __future__ import annotations
from loguru import logger

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    SparseEmbedding,
    ImageEmbedding,
    LateInteractionTextEmbedding,
    LateInteractionMultimodalEmbedding,
)

from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class FastEmbedEmbeddingModelBuilder:
    @staticmethod
    def build_supported_text_embedding_model(
        model_name: str,
        specific_model_path: str | Path,
    ) -> TextEmbedding:
        text_embedding_model = TextEmbedding(
            model_name=model_name,
            # cache_dir=,
            specific_model_path=str(specific_model_path),
            # HARDCODED
            local_files_only=True,
        )
        return text_embedding_model

