"""
Sources:
    https://github.com/yuliu625/Yu-AI-Inference-Manifests/model_management_cli.py

References:
    https://github.com/yuliu625/Yu-AI-Inference-Stack/model_foundations/transformers/hf_download.py

Synopsis:
    huggingface 仓库统一模型管理的 CLI 。

Notes:
    使用配置文件下载模型。
    在原有 downloader 的基础上，对配置文件参数进行校验，并以 CLI 执行下载。
"""

from __future__ import annotations
from loguru import logger

from model_foundations.hf_downloader import HFDownloader
from pydantic import BaseModel, Field
from pathlib import Path
import yaml
import typer

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class HFDownloaderConfigs(BaseModel):
    """
    All configs.
    """
    class Settings(BaseModel):
        """
        Downloader configs.
        """
        local_model_dir: str
        local_dataset_dir: str
        cache_dir: str | None
        is_only_torch: bool
        repo_concurrency: int
        file_concurrency: int
        hf_endpoint: str | None
        hf_home: str | None

    class Tasks(BaseModel):
        """
        Downloading tasks.
        """
        models: list[str] = Field(
            default_factory=list,
        )
        datasets: list[str] = Field(
            default_factory=list,
        )

    settings: Settings
    tasks: Tasks


app = typer.Typer(
    rich_markup_mode='rich',
)


@app.command()
def download_models(
    config_path: str = typer.Option(
        "config.toml", "--config", "-c", help="config file path"
    )
):
    """
    Huggingface downloader CLI.
    """
    # path processing
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not exist: {config_path}")
        raise typer.Exit(code=1)
    # read config file
    raw_configs = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    configs = HFDownloaderConfigs(**raw_configs)
    # init downloader
    downloader = HFDownloader(
        local_model_dir=configs.settings.local_model_dir,
        local_dataset_dir=configs.settings.local_dataset_dir,
        cache_dir=configs.settings.cache_dir,
        is_only_torch=configs.settings.is_only_torch,
        repo_concurrency=configs.settings.repo_concurrency,
        file_concurrency=configs.settings.file_concurrency,
        hf_endpoint=configs.settings.hf_endpoint,
        hf_home=configs.settings.hf_home,
    )
    # run downloading tasks
    logger.info(f"Downloading models: {configs.tasks.models}")
    downloader.download_models(configs.tasks.models)
    logger.success(f"Finish model tasks.")


@app.command()
def download_models(
    config_path: str = typer.Option(
        "config.toml", "--config", "-c", help="config file path"
    )
):
    """
    Huggingface downloader CLI.
    """
    # path processing
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not exist: {config_path}")
        raise typer.Exit(code=1)
    # read config file
    raw_configs = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    configs = HFDownloaderConfigs(**raw_configs)
    # init downloader
    downloader = HFDownloader(
        local_model_dir=configs.settings.local_model_dir,
        local_dataset_dir=configs.settings.local_dataset_dir,
        cache_dir=configs.settings.cache_dir,
        is_only_torch=configs.settings.is_only_torch,
        repo_concurrency=configs.settings.repo_concurrency,
        file_concurrency=configs.settings.file_concurrency,
        hf_endpoint=configs.settings.hf_endpoint,
        hf_home=configs.settings.hf_home,
    )
    # run downloading tasks
    logger.info(f"Downloading datasets: {configs.tasks.models}")
    downloader.download_datasets(configs.tasks.datasets)
    logger.success(f"Finish dataset tasks.")


if __name__ == '__main__':
    app()

