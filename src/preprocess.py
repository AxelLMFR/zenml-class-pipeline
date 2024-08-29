import pandas as pd
from config import Config
from loguru import logger


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

    def load_data(self) -> None:
        """Loads the data."""
        self.data = pd.read_csv(self.config.artefact_path.PREPROCESS_INPUT_DATA)
        logger.info("Data loaded.")

    def drop_duplicates(self) -> None:
        """Drops duplicates."""
        self.preprocessed_data = self.data.drop_duplicates()
        logger.info("Duplicates dropped.")

    def rename_columns(self) -> None:
        """Renames columns."""
        self.preprocessed_data = self.preprocessed_data.rename(
            columns=dict(
                zip(
                    self.config.preprocess.OLD_COLUMN_NAMES,
                    self.config.preprocess.NEW_COLUMN_NAMES,
                )
            )
        )
        logger.info("Columns renamed.")

    def save_data(self) -> None:
        """Saves the preprocessed data."""
        self.preprocessed_data.to_csv(
            self.config.artefact_path.PREPROCESS_OUTPUT_DATA, index=False
        )
        logger.info("Data saved.")

    def category_encode(self) -> None:
        """Encodes the target column."""
        self.preprocessed_data["y"] = (
            self.preprocessed_data["y"].astype("category").cat.codes
        )
        logger.info("Target column encoded.")

    def main(self) -> None:
        """Runs the preprocessing pipeline."""
        self.load_data()
        self.drop_duplicates()
        self.rename_columns()
        self.category_encode()
        self.save_data()
