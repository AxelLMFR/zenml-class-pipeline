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

    def load_data(self):
        """Load the data using tf.data."""
        dataset = tf.data.experimental.make_csv_dataset(
            self.config.artefact_path.TRAIN_INPUT_DATA,
            batch_size=5,
            column_names=self.config.preprocess.NEW_COLUMN_NAMES,
            label_name=self.config.train.TARGET_COLUMN_NAME,
            num_epochs=1,
            shuffle=False,
        )

        # Rename the input features to match the expected input layer name
        def rename_features(features, label):
            return {'input_layer': tf.stack(list(features.values()), axis=1)}, label

        self.data = dataset.map(rename_features)
        logger.info("Data loaded.")

    def create_model(self):
        """Create a simple model."""
        inputs = tf.keras.Input(shape=(4,), name='input_layer')
        layer = tf.keras.layers.Dense(10, activation="relu")(inputs)
        output = tf.keras.layers.Dense(3, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        logger.info("Model created.")

    def train_model(self):
        """Train the model."""
        self.model.fit(self.data, epochs=10)
        logger.info("Model trained.")

    def evaluate_model(self):
        """Evaluate the model."""
        loss, accuracy = self.model.evaluate(self.data)
        logger.info(f"Loss: {loss}, Accuracy: {accuracy}")

    def save_model(self):
        """Save the model."""
        self.model.save(self.config.artefact_path.TRAIN_MODEL)
        logger.info("Model saved.")

    def main(self):
        """Run the training pipeline."""
        self.load_data()
        self.create_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()