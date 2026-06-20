"""
对于前沿模型 custom serving 的暂时适配。

Notes:
    部分模型因新或存在特殊独立方法，未被 inference_engines 的方法支持。
    因此，使用 litserve 对于纯模型推理进行封装。
"""

