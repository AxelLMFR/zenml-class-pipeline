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

    class Config:
        case_sensitive = True


class Config(BaseSettings):
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    artefact_path: ArtefactPathConfig = Field(default_factory=ArtefactPathConfig)
