"""
Launch bge m3 embedding model.
"""

from __future__ import annotations
import pytest
from loguru import logger

import litserve as ls
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING, Literal, Sequence
# if TYPE_CHECKING:


class EmbeddingRequest(BaseModel):
    """
    对于 bge-m3 的 all-in-one 请求 schema 。
    """
    embedding_type: Literal['query', 'text'] = Field(
        ...,
        description="编码类型，bge-m3 需要区别 query 和 text 。",
    )
    text: str = Field(
        ...,
        description="需要被编码的文本。",
    )


class QueryEmbeddingRequest(BaseModel):
    text: str = Field(
        ...,
        description="需要被编码的 query 。",
    )


class TextEmbeddingRequest(BaseModel):
    text: str = Field(
        ...,
        description="需要被编码的 text 。",
    )


class EmbeddingResponse(BaseModel):
    """
    encoding results
    """
    dense: list
    sparse: list
    multi_vector: list


class BGEM3EmbeddingModelQueryLitAPI(ls.LitAPI):
    """
    将 dense, sparse, multi-vector 方法全部封装，全部计算。

    使用原本已经可运行的封装类，未进行任何额外优化。
    """
    def setup(self, device):
        self._model = BGEM3FlagModel(
            model_name_or_path=r"D:\model\BAAI\bge-m3",
            # HARDCODED
            use_fp16=True,
        )
        self._batch_size = 8
        # HARDCODED
        ## 使用最大 context length 能力。
        self._max_length = 8192
        ## 测试使用，所有的 encode 方法的结果都返回。
        self._return_dense = True
        self._return_sparse = True
        self._return_colbert_vecs = True

    def decode_request(self, request, **kwargs):
        return request['text']

    def predict(self, query: str, **kwargs):
        # encode queries in different methods
        results = self._model.encode_queries(
            queries=[query],
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        processed_result = self.process_raw_results(
            raw_results=results,
        )
        return dict(
            dense=processed_result['dense'][0],
            sparse=processed_result['sparse'][0],
            multi_vector=processed_result['multi_vector'][0],
        )

    def encode_response(self, output, **kwargs):
        return output
        return EmbeddingResponse(
            dense=output['dense'],
            sparse=output['sparse'],
            multi_vector=output['multi_vector'],
        )

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
        sparse = []
        for doc_weights in raw_results['lexical_weights']:
            # 将字典中的 numpy.float32 转换为原生 float
            converted_weights = {
                str(token_id): float(weight)
                for token_id, weight in doc_weights.items()
            }
            sparse.append(converted_weights)
        processed_result = dict(
            dense=raw_results['dense_vecs'].tolist(),
            sparse=sparse,
            multi_vector=[
                np_multi_vector.tolist()
                for np_multi_vector in raw_results['colbert_vecs']
            ],
        )
        assert isinstance(processed_result['dense'], list)
        assert isinstance(processed_result['sparse'], list)
        assert isinstance(processed_result['multi_vector'], list)
        return processed_result


class BGEM3EmbeddingModelTextLitAPI(ls.LitAPI):
    """
    将 dense, sparse, multi-vector 方法全部封装，全部计算。

    使用原本已经可运行的封装类，未进行任何额外优化。
    """
    def setup(self, device):
        self._model = BGEM3FlagModel(
            model_name_or_path=r"D:\model\BAAI\bge-m3",
            # HARDCODED
            use_fp16=True,
        )
        self._batch_size = 8
        # HARDCODED
        ## 使用最大 context length 能力。
        self._max_length = 8192
        ## 测试使用，所有的 encode 方法的结果都返回。
        self._return_dense = True
        self._return_sparse = True
        self._return_colbert_vecs = True

    async def decode_request(self, request, **kwargs):
        return request['text']

    async def predict(self, x: str, **kwargs):
        # encode texts in different methods
        results = self._model.encode(
            sentences=[x],
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        processed_result = self.process_raw_results(
            raw_results=results,
        )
        return dict(
            dense=processed_result['dense'][0],
            sparse=processed_result['sparse'][0],
            multi_vector=processed_result['multi_vector'][0],
        )

    async def encode_response(self, output, **kwargs):
        return output
        return EmbeddingResponse(
            dense=output['dense'],
            sparse=output['sparse'],
            multi_vector=output['multi_vector'],
        )

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


if __name__ == '__main__':
    lit_api = BGEM3EmbeddingModelQueryLitAPI(
        # max_batch_size=1,
        # api_path='/predict',
    )
    server = ls.LitServer(
        lit_api=lit_api,
        accelerator='auto',
    )
    # start server
    server.run(
        # host='0.0.0.0',
        # port=8000,
    )

