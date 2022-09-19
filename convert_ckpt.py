from typer import Argument, Option, Typer

from tsdae import KoTSDAEModule

cli = Typer()


@cli.command(no_args_is_help=True)
def main(
    path: str = Argument(..., help="변경할 체크포인트파일의 경로"),
    output: str = Option("output", help="출력 경로"),
):
    module = KoTSDAEModule.load_from_checkpoint(path)
    module.save(output)


if __name__ == "__main__":
    cli()
