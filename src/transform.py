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
    """

    def __init__(self) -> None:
        """Constructor."""
        self.config = Config()
        logger.info("Transform class initialized")

    def load_data(self):
        """Load the data using tf.data."""
        dataset = tf.data.experimental.make_csv_dataset(
            self.config.artefact_path.TRANSFORM_INPUT_DATA,
            batch_size=2,
            column_names=self.config.preprocess.NEW_COLUMN_NAMES[:-1], # Exclude the target column
            num_epochs=1,
            shuffle=False,
        )
        def rename_features(features):
            return {'input_layer': tf.stack(list(features.values()), axis=1)}

        self.data = dataset.map(rename_features)
        logger.info("Data loaded.")

    def prepare_data(self):
        """Prepare the data to concatenate the features into input_layer."""
        self.data = next(iter(self.data))
        self.data = tf.concat(list(self.data.values()), axis=1)
        logger.info("Data prepared.")

    def load_model(self):
        """Load the trained model."""
        self.model = tf.keras.models.load_model(self.config.artefact_path.TRAIN_MODEL)
        logger.info("Model loaded.")

    def transform_data(self):
        """Transform the data."""
        self.predictions = self.model.predict(self.data)
        logger.info("Data transformed.")

    def save_data(self):
        """Save the predictions."""
        pd.DataFrame(self.predictions).to_csv(self.config.artefact_path.TRANSFORM_OUTPUT_DATA, index=False)
        logger.info("Data saved.")

    def main(self):
        """Run the main pipeline."""
        self.load_data()
        self.prepare_data()
        self.load_model()
        self.transform_data()
        self.save_data()
        logger.info("Transform pipeline complete.")
