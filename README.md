# zenml-class-pipeline

## Architecture

### data

This is where the raw data is stored. In this case, we are using the `iris` dataset from `sklearn`.

### models

This is where the trained model is stored.

### src

This is where the code for the pipeline is stored. The `main.py` file is the entry point for the pipeline.

## How to execute this pipeline

```shell
poetry install
poetry shell
```

to run the preprocessing step:

```shell
python src/main.py preprocess
```

to run the training step:

```shell
python src/main.py train
```

to run the transform step:

```shell
python src/main.py transform
```

Or to run the full pipeline:

```shell
python src/main.py pipeline
```
