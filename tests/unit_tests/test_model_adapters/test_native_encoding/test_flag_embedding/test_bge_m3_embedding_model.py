"""
Tests for bge-m3 embedding model.
"""

from __future__ import annotations
import pytest
from loguru import logger

from native_encoding.flag_embedding.bge_m3_embedding_model import (
    BGEM3EmbeddingModel,
)

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


def _get_model_path():
    return r"D:\model\BAAI\bge-m3"


class TestBgeM3EmbeddingModel:
    @pytest.mark.parametrize(
        'model_name_or_path', [
        _get_model_path()
    ],)
    def test_encode_texts(
        self,
        model_name_or_path: str,
    ):
        bge_m3_embedding_model = BGEM3EmbeddingModel(
            model_name_or_path=model_name_or_path,
            batch_size=1,
        )
        results = bge_m3_embedding_model.encode_texts(
            texts=['haha', 'xixi',]
        )
        logger.info(f"\nTexts Results: \n{results}")

    @pytest.mark.parametrize(
        'model_name_or_path', [
        _get_model_path()
    ],)
    def test_encode_queries(
        self,
        model_name_or_path: str,
    ):
        bge_m3_embedding_model = BGEM3EmbeddingModel(
            model_name_or_path=model_name_or_path,
            batch_size=1,
        )
        results = bge_m3_embedding_model.encode_queries(
            queries=['haha', 'xixi',]
        )
        logger.info(f"\nQueries Results: \n{results}")

    @pytest.mark.parametrize(
        'model_name_or_path', [
        _get_model_path()
    ],)
    def test_encode_query(
        self,
        model_name_or_path: str,
    ):
        bge_m3_embedding_model = BGEM3EmbeddingModel(
            model_name_or_path=model_name_or_path,
            batch_size=1,
        )
        results = bge_m3_embedding_model.encode_query(
            query='haha',
        )
        logger.info(f"\nQuery Results: \n{results}")

