import tensorflow as tf
from config import Config
from loguru import logger


class Train:
    """Training pipeline.

    Attributes:
        config (Config): The configuration object.
        data (tf.data.Dataset): The input data.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.config = Config()
        logger.info("BigQuery client created.")

    def load_data(self) -> None:
        """Loads the data using tf.data."""
        dataset = tf.data.experimental.make_csv_dataset(
            self.config.artefact_path.TRAIN_INPUT_DATA,
            batch_size=self.config.train.BATCH_SIZE,
            column_names=self.config.preprocess.NEW_COLUMN_NAMES,
            label_name=self.config.train.TARGET_COLUMN_NAME,
            num_epochs=1,  # To read the data only once
            shuffle=False,
        )

        # Rename the input features to match the expected input layer name
        def rename_features(features: dict[str, tf.Tensor], label: tf.Tensor) -> tuple:
            """Renames the features.

            Args:
                features (dict[str, tf.Tensor]): The features.
                label (tf.Tensor): The label.

            Returns:
                tuple: The renamed features and the label.
            """
            return {
                self.config.train.INPUT_LAYER_NAME: tf.stack(
                    list(features.values()), axis=1
                )
            }, label

        self.data = dataset.map(rename_features)
        logger.info("Data loaded.")

    def create_model(self) -> None:
        """Creates a simple model."""
        inputs = tf.keras.Input(
            shape=self.config.train.INPUT_SHAPE, name=self.config.train.INPUT_LAYER_NAME
        )
        layer = tf.keras.layers.Dense(
            self.config.train.DENSE_UNITS, activation=self.config.train.DENSE_ACTIVATION
        )(inputs)
        output = tf.keras.layers.Dense(
            self.config.train.OUTPUT_UNITS,
            activation=self.config.train.OUTPUT_ACTIVATION,
        )(layer)
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        logger.info("Model created.")

    def train_model(self) -> None:
        """Trains the model."""
        self.model.fit(self.data, epochs=self.config.train.EPOCHS)
        logger.info("Model trained.")

    def evaluate_model(self) -> None:
        """Evaluates the model."""
        loss, accuracy = self.model.evaluate(self.data)
        logger.info(f"Loss: {loss}, Accuracy: {accuracy}")

    def save_model(self) -> None:
        """Saves the model."""
        self.model.save(self.config.artefact_path.TRAIN_MODEL)
        logger.info("Model saved.")

    def main(self) -> None:
        """Run the training pipeline."""
        self.load_data()
        self.create_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
