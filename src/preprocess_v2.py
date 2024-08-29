import pandas as pd
from config import Config
from loguru import logger
from typing import Type, Dict, Any, List
from zenml.steps import BaseStep
from zenml import pipeline


class Preprocess:
    """Preprocessing pipeline.

    Attributes:
        config (Config): The configuration object.
        data (pd.DataFrame): The input data.
        preprocessed_data (pd.DataFrame): The preprocessed data.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.config = Config()
        self.data = None
        self.preprocessed_data = None
        logger.info("Preprocess class initialized.")

    def load_data(self) -> pd.DataFrame:
        """Loads the data."""
        data = pd.read_csv(self.config.artefact_path.PREPROCESS_INPUT_DATA)
        logger.info("Data loaded.")
        return data

    def drop_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicates."""
        preprocessed_data = data.drop_duplicates()
        logger.info("Duplicates dropped.")
        return preprocessed_data

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Renames columns."""
        preprocessed_data = data.rename(
            columns=dict(
                zip(
                    self.config.preprocess.OLD_COLUMN_NAMES,
                    self.config.preprocess.NEW_COLUMN_NAMES,
                )
            )
        )
        logger.info("Columns renamed.")
        return preprocessed_data

    def save_as_csv(self, data: pd.DataFrame) -> str:
        """Saves the preprocessed data."""
        csv_data = data.to_csv(index=False)
        return csv_data

    def category_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encodes the target column."""
        data["y"] = data["y"].astype("category").cat.codes
        logger.info("Target column encoded.")
        return data

    # def main(self) -> None:
    #     """Runs the preprocessing pipeline."""
    #     self.load_data()
    #     self.drop_duplicates()
    #     self.rename_columns()
    #     self.category_encode()
    #     self.save_data()


def extract_steps(obj: Any, method_names: List[str]) -> Dict[str, BaseStep]:
    step_instances = {}
    for method_name in method_names:
        method = getattr(obj, method_name)
        class_name = f"{method_name}_step"

        step_class: Type["BaseStep"] = type(
            class_name,
            (BaseStep,),
            {
                "entrypoint": staticmethod(method),
                "__module__": method.__module__,
                "__doc__": method.__doc__,
                "source_object": property(lambda _: method),
            },
        )

        globals()[class_name] = step_class
        step_instances[method_name] = step_class()

    return step_instances


steps = extract_steps(
    Preprocess(),
    method_names=[
        "load_data",
        "drop_duplicates",
        "rename_columns",
        "category_encode",
        "save_as_csv",
    ],
)
load_data, drop_duplicates, rename_columns, category_encode, save_as_csv = (
    steps.values()
)


@pipeline(enable_cache=False)
def preprocess_pipeline():
    data = load_data()
    data = drop_duplicates(data)
    data = rename_columns(data)
    data = category_encode(data)
    save_as_csv(data)


if __name__ == "__main__":
    preprocess_pipeline()
