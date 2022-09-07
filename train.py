import os
from textwrap import dedent
from typing import Optional

import pytorch_lightning as pl
import typer
import yaml
from datasets import load_dataset
from loguru import logger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from typer import Argument, Option, Typer

from tsdae import KoDenoisingAutoEncoderDataset, KoTSDAEModule

cli = Typer(name="tsdae", pretty_exceptions_show_locals=False)


def dedent_text(text: str) -> str:
    return dedent(text).strip().replace("\n", " ")


decoder_name_desc = """
    디코더로 사용할 모델의 이름 또는 경로,
    훈련할 모델을 AutoModelForCausalLM으로 불러올 수 있고,
    해당 AutoModelForCausalLM이 `encoder_hidden_states`를
    입력으로 받을 수 있다면 `None`으로 설정하십시오.
    이 링크에서 확인:
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
    """
decoder_name_desc = dedent_text(decoder_name_desc)

max_seq_length_desc = """
    모델이 입력으로 받을 수 있는 최대 길이,
    `CUDA error: device-side assert triggered` 에러 발생시
    수동으로 조절해보십시오.
    """
max_seq_length_desc = dedent_text(max_seq_length_desc)


def config_callback(
    ctx: typer.Context, param: typer.CallbackParam, value: Optional[str]
):
    "https://github.com/tiangolo/typer/issues/86#issuecomment-996374166"
    if value:
        typer.echo(f"Loading config file: {value}")
        try:
            with open(value) as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex)) from ex
    return value


@cli.command(no_args_is_help=True)
def main(
    model_name_or_path: str = Argument(
        ...,
        help="사용할 모델의 huggingface 이름, 또는 경로",
        rich_help_panel="모델",
        show_default=False,
    ),
    dataset_name: str = Argument(
        ..., help="사용할 데이터셋의 huggingface 이름", rich_help_panel="데이터", show_default=False
    ),
    max_seq_length: Optional[int] = Option(
        None, help=max_seq_length_desc, rich_help_panel="모델"
    ),
    config: Optional[str] = Option(
        None, help="설정을 담은 yaml 파일 경로", callback=config_callback, is_eager=True
    ),
    optimizer_name: str = Option(
        "adamp",
        help="사용할 옵티마이저의 이름, pytorch_optimizer에서 지원하는 옵티마이저",
        rich_help_panel="훈련",
    ),
    lr: float = Option(5e-5, help="Learning rate", rich_help_panel="훈련"),
    weight_decay: float = Option(
        0.0, help="Weight decay", min=0.0, max=1.0, rich_help_panel="훈련"
    ),
    batch_size: int = Option(8, help="Batch size", min=1, rich_help_panel="훈련"),
    max_steps: int = Option(1_000_000, help="훈련 스텝 수", rich_help_panel="훈련"),
    gradient_clip_val: Optional[float] = Option(
        None, help="Gradient clipping", min=0.0, rich_help_panel="훈련"
    ),
    accumulate_grad_batches: Optional[int] = Option(
        None, help="Gradient accumulation", rich_help_panel="훈련"
    ),
    decoder_name: Optional[str] = Option(
        None, help=decoder_name_desc, rich_help_panel="모델"
    ),
    dataset_name2: Optional[str] = Option(
        None, help="load_dataset의 두 번째 인자로 들어갈 이름", rich_help_panel="데이터"
    ),
    dataset_split: str = Option(
        "train", help="데이터셋에서 사용할 split", rich_help_panel="데이터"
    ),
    text_col: str = Option("text", help="데이터셋에서 text를 담은 열의 이름", rich_help_panel="데이터"),
    use_auth_token: Optional[bool] = Option(
        None, help="huggingface auth token", rich_help_panel="데이터"
    ),
    num_workers: int = Option(
        8, help="데이터 로더에서 사용할 프로세스 수, windows면 0으로 고정됨", rich_help_panel="훈련"
    ),
    fast_dev_run: bool = Option(False, help="훈련 테스트를 실행합니다.", rich_help_panel="훈련"),
    output_path: Optional[str] = Option(None, help="모델을 저장할 경로", rich_help_panel="훈련"),
    save_steps: int = Option(10_000, help="모델을 저장할 주기", rich_help_panel="훈련"),
    wandb_name: Optional[str] = Option(None, help="wandb 이름", rich_help_panel="훈련"),
    log_every_n_steps: int = Option(200, help="몇 스텝마다 로그를 남길지", rich_help_panel="훈련"),
    seed: int = Option(42, help="랜덤 시드", rich_help_panel="훈련"),
):
    # 모델
    module = KoTSDAEModule(
        model=model_name_or_path,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        decoder_name=decoder_name,
        max_seq_length=max_seq_length,
    )
    logger.debug("모델 생성 완료")

    # 데이터셋
    hf_dataset = load_dataset(
        dataset_name, dataset_name2, use_auth_token=use_auth_token, split=dataset_split
    )
    dataset = KoDenoisingAutoEncoderDataset(hf_dataset, text_col)
    logger.debug("데이터셋 생성 완료")

    if os.name == "nt":
        logger.warning("윈도우에서는 num_workers를 0으로 고정합니다.")
        num_workers = 0

    # 데이터로더
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=module.model.smart_batching_collate,
        num_workers=num_workers,
    )

    # 훈련
    if output_path is None:
        output_path = "result"
    logger.info(f"훈련 결과를 {output_path}에 저장합니다.")

    if wandb_name is None:
        wandb_name = f"{model_name_or_path}-{dataset_name}"

    wandb_logger = WandbLogger(name=wandb_name, project="tsdae")
    wandb_logger.watch(module)

    pl.seed_everything(seed)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(refresh_rate=10),
        ModelCheckpoint(dirpath="checkpoints", every_n_train_steps=save_steps),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        precision=16,
        log_every_n_steps=log_every_n_steps,
        fast_dev_run=fast_dev_run,
    )
    logger.debug(f"훈련 시작, 총 스텝: {trainer.estimated_stepping_batches}")
    trainer.fit(module, train_dataloaders=train_loader)

    logger.debug("훈련 종료")
    module.model.save(output_path)
    logger.debug("모델 저장 완료")


if __name__ == "__main__":
    cli()
