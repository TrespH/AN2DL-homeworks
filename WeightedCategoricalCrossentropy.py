from tensorflow.keras.losses import Loss
import keras
import tensorflow as tf
from tensorflow.keras import backend as K


@keras.saving.register_keras_serializable()
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, name="weighted_categorical_crossentropy"):
        """
        Custom loss function for weighted categorical crossentropy.

        Args:
        - class_weights (list or tensor, optional): A list or tensor of weights for each class.
        - name (str, optional): Name of the loss function.
        """
        super(WeightedCategoricalCrossentropy, self).__init__(name=name)

        if class_weights is not None:
            self.class_weights = tf.convert_to_tensor(
                class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

    def call(self, y_true, y_pred):
        """
        Computes the weighted categorical crossentropy between true labels and predictions.

        Args:
        - y_true: Ground truth labels.
        - y_pred: Predicted probabilities.

        Returns:
        - loss: Weighted categorical crossentropy loss value.
        """
        # Clip predictions to prevent log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Compute the categorical crossentropy
        crossentropy = -K.sum(y_true * K.log(y_pred), axis=-1)

        if self.class_weights is not None:
            # Apply the class weights
            weights = K.sum(self.class_weights * y_true, axis=-1)
            loss = crossentropy * weights
        else:
            loss = crossentropy

        return K.mean(loss)


# Define class weights (e.g., higher for hard classes)
class_weights = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.5, 1.0]
loss_fn = WeightedCategoricalCrossentropy(class_weights=class_weights)
