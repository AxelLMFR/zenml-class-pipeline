# zenml-class-pipeline

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

to run the full pipeline:

```shell
python src/main.py pipeline
```
