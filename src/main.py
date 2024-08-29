import click
from preprocess import Preprocess
from train import Train
from transform import Transform


@click.group()
def cli() -> None:
    """CLI for the model."""
    pass


@click.command()
def preprocess() -> None:
    """Preprocess the data."""
    p = Preprocess()
    p.main()


@click.command()
def train() -> None:
    """Train the model."""
    t = Train()
    t.main()


@click.command()
def transform() -> None:
    """Evaluate the model."""
    t = Transform()
    t.main()


@click.command()
def pipeline() -> None:
    """Run the full pipeline."""
    p = Preprocess()
    p.main()
    t = Train()
    t.main()
    t = Transform()
    t.main()


if __name__ == "__main__":
    cli.add_command(preprocess)
    cli.add_command(train)
    cli.add_command(transform)
    cli.add_command(pipeline)
    cli()
