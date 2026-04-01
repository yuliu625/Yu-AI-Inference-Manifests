"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/native_encoding/litserve/bge_m3_embedding_model.py

References:
    https://huggingface.co/BAAI/bge-m3

Synopsis:
    将 FlagEmbedding 的 bge-m3 使用 LitServe 推理服务化。

Notes:
    因为 FlagEmbedding 设计的问题，每次导入速度慢且不稳定。
    这里通过 litserve 进行服务化，并稳定推理过程。
"""

from __future__ import annotations
from loguru import logger

import litserve as ls
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING, Literal, Sequence
# if TYPE_CHECKING:


class EmbeddingRequest(BaseModel):
    text: str = Field(
        description="需要被编码的文本。",
    )
    embedding_type: Literal['query', 'text'] = Field(
        description="编码类型，bge-m3 需要区别 query 和 text 。",
    )


class BGEM3EmbeddingModelLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = BGEM3FlagModel(
            model_name_or_path=r"",
            # HARDCODED
            use_fp16=True,
        )

    async def decode_request(self, request):
        ...

    async def predict(self, x):
        ...

    async def encode_response(self, output):
        ...

