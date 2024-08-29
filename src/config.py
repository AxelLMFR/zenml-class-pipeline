from pydantic import Field
from pydantic_settings import BaseSettings


class ArtefactPathConfig(BaseSettings):
    PREPROCESS_INPUT_DATA: str = Field(
        default="data/iris.csv",
        description="Path to the input data.",
        env="PATH_PREPROCESS_INPUT_DATA",
    )
    PREPROCESS_OUTPUT_DATA: str = Field(
        default="data/iris_preprocessed.csv",
        description="Path to the preprocessed data.",
        env="PATH_PREPROCESS_OUTPUT_DATA",
    )
    TRAIN_INPUT_DATA: str = Field(
        default="data/iris_preprocessed.csv",
        description="Path to the preprocessed data.",
        env="PATH_TRAIN_INPUT_DATA",
    )

    TRAIN_MODEL: str = Field(
        default="models/iris_model.h5",
        description="Path to the save the trained model.",
        env="PATH_TRAIN_MODEL",
    )

    TRANSFORM_INPUT_DATA: str = Field(
        default="data/to_predict.csv",
        description="Path to the input data.",
        env="PATH_TRANSFORM_INPUT_DATA",
    )

    TRANSFORM_OUTPUT_DATA: str = Field(
        default="data/predictions.csv",
        description="Path to the predictions.",
        env="PATH_TRANSFORM_OUTPUT_DATA",
    )

    class Config:
        case_sensitive = True


class PreprocessConfig(BaseSettings):
    OLD_COLUMN_NAMES: list[str] = Field(
        default=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "variety",
        ],
        description="List of old column names.",
        env="PREPROCESS_OLD_COLUMN_NAMES",
    )

    NEW_COLUMN_NAMES: list[str] = Field(
        default=["sepal_length", "sepal_width", "petal_length", "petal_width", "y"],
        description="List of new column names.",
        env="PREPROCESS_NEW_COLUMN_NAMES",
    )

    class Config:
        case_sensitive = True


class TrainConfig(BaseSettings):
    TARGET_COLUMN_NAME: str = Field(
        default="y",
        description="Target column name.",
        env="TRAIN_TARGET_COLUMN_NAME",
    )

    BATCH_SIZE: int = Field(
        default=5,
        description="Batch size.",
        env="TRAIN_BATCH_SIZE",
    )

    INPUT_SHAPE: tuple[int] = Field(
        default=(4,),
        description="Input shape.",
        env="TRAIN_INPUT_SHAPE",
    )

    INPUT_LAYER_NAME: str = Field(
        default="input_layer",
        description="Input layer name.",
        env="TRAIN_INPUT_LAYER_NAME",
    )

    DENSE_UNITS: int = Field(
        default=10,
        description="Number of units in the dense layer.",
        env="TRAIN_DENSE_UNITS",
    )

    DENSE_ACTIVATION: str = Field(
        default="relu",
        description="Activation function for the dense layer.",
        env="TRAIN_DENSE_ACTIVATION",
    )

    OUTPUT_UNITS: int = Field(
        default=3,
        description="Number of output units.",
        env="TRAIN_OUTPUT_UNITS",
    )

    OUTPUT_ACTIVATION: str = Field(
        default="softmax",
        description="Activation function for the output layer.",
        env="TRAIN_OUTPUT_ACTIVATION",
    )

    EPOCHS: int = Field(
        default=10,
        description="Number of epochs.",
        env="TRAIN_EPOCHS",
    )

    class Config:
        case_sensitive = True


class TransformConfig(BaseSettings):

    BATCH_SIZE: int = Field(
        default=2,
        description="Batch size.",
        env="TRANSFORM_BATCH_SIZE",
    )

    class Config:
        case_sensitive = True

class Config(BaseSettings):
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    artefact_path: ArtefactPathConfig = Field(default_factory=ArtefactPathConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)

