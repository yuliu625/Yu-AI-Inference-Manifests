"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/native_encoding/flag_embedding/bge_m3_embedding_model.py

References:
    https://huggingface.co/BAAI/bge-m3

Synopsis:
    基于 FlagEmbedding 的 bge-m3 直接使用方法。

Notes:
    由于 bge-m3 有多种编码和检索方法，使用各种 all-in-one 需要基于 FlagEmbedding 才能使用。

    注意:
        - 以下实现实际上是一个 wrapper ，硬编码了 bge-m3 科研场景的必要参数。
        - 如果有进阶定制需求，应直接使用 BGEM3FlagModel 这个在 transformers 上的直接 wrapper 。
"""

from __future__ import annotations
from loguru import logger

from FlagEmbedding import BGEM3FlagModel

from typing import TYPE_CHECKING, Sequence
# if TYPE_CHECKING:


class BGEM3EmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int,
    ):
        self._model = BGEM3FlagModel(
            model_name_or_path=model_name_or_path,
            # HARDCODED
            use_fp16=True,
        )
        self._batch_size = batch_size
        # HARDCODED
        ## 使用最大 context length 能力。
        self._max_length = 8192
        ## 测试使用，所有的 encode 方法的结果都返回。
        self._return_dense = True
        self._return_sparse = True
        self._return_colbert_vecs = True

    def encode_query(
        self,
        query: str,
    ) -> dict:
        # encode queries in different methods
        results = self._model.encode_queries(
            queries=[query],
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        processed_result = self.process_raw_results(raw_results=results)
        return dict(
            dense=processed_result['dense'][0],
            sparse=processed_result['sparse'][0],
            multi_vector=processed_result['multi_vector'][0],
        )

    def encode_queries(
        self,
        queries: Sequence[str],
    ) -> dict:
        # encode queries in different methods
        results = self._model.encode_queries(
            queries=list(queries),
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        processed_result = self.process_raw_results(
            raw_results=results,
        )
        return processed_result

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> dict:
        # encode texts in different methods
        results = self._model.encode(
            sentences=list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        processed_result = self.process_raw_results(
            raw_results=results,
        )
        return processed_result

    def process_raw_results(
        self,
        raw_results: dict,
    ) -> dict[str, list]:
        """
        将 FlagEmbedding 返回的结果进行转换，统一为 python object 。

        执行 2 项操作:
            - np to list: 将格式不一致的 np 转换为统一的多维 list 。
            - name mapping: 将晦涩和简写的 keys 进行统一转换。

        Args:
            raw_results (dict): FlagEmbedding 返回的原始结果。

        Returns:
            dict[str, list]: 转换后的结果。
        """
        processed_result = dict(
            dense=raw_results['dense_vecs'].tolist(),
            sparse=raw_results['lexical_weights'],
            multi_vector=[
                np_multi_vector.tolist()
                for np_multi_vector in raw_results['colbert_vecs']
            ],
        )
        assert isinstance(processed_result['dense'], list)
        assert isinstance(processed_result['sparse'], list)
        assert isinstance(processed_result['multi_vector'], list)
        return processed_result

