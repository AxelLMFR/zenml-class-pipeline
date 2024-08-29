import tensorflow as tf
from config import Config
from loguru import logger
import pandas as pd


class Transform:
    """Transform pipeline.

    Attributes:
        config (Config): The configuration object.
        data (tf.data.Dataset): The input data.
        model (tf.keras.Model): The trained model.
        predictions (tf.Tensor): The predictions.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.config = Config()
        self.data = None
        self.model = None
        self.predictions = None
        logger.info("Transform class initialized")

    def load_data(self) -> None:
        """Loads the data using tf.data."""
        dataset = tf.data.experimental.make_csv_dataset(
            self.config.artefact_path.TRANSFORM_INPUT_DATA,
            batch_size=self.config.transform.BATCH_SIZE,
            column_names=self.config.preprocess.NEW_COLUMN_NAMES[
                :-1
            ],  # Exclude the target column
            num_epochs=1,
            shuffle=False,
        )

        def rename_features(features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
            """Renames the features.

            Args:
                features (dict[str, tf.Tensor]): The features.

            Returns:
                dict[str, tf.Tensor]: The renamed features.
            """
            return {
                self.config.train.INPUT_LAYER_NAME: tf.stack(
                    list(features.values()), axis=1
                )
            }

        self.data = dataset.map(rename_features)
        logger.info("Data loaded.")

    def load_model(self) -> None:
        """Loads the trained model."""
        self.model = tf.keras.models.load_model(self.config.artefact_path.TRAIN_MODEL)
        logger.info("Model loaded.")

    def transform_data(self) -> None:
        """Transforms the data."""
        self.predictions = self.model.predict(self.data)
        logger.info("Data transformed.")

    def save_data(self) -> None:
        """Saves the predictions."""
        pd.DataFrame(self.predictions).to_csv(
            self.config.artefact_path.TRANSFORM_OUTPUT_DATA, index=False
        )
        logger.info("Data saved.")

    def main(self) -> None:
        """Runs the main pipeline."""
        self.load_data()
        self.load_model()
        self.transform_data()
        self.save_data()
        logger.info("Transform pipeline complete.")
