import numpy as np
import keras as tfk
import cv2  

class Model:
    def __init__(self):
        """
        Initialize the internal state of the model. Note that the __init__
        method cannot accept any arguments.

        The following is an example loading the weights of a pre-trained
        model.
        """
        self.neural_network = tfk.models.load_model('mnV2_lion_resize.keras')

    def predict(self, X):
        """
        Predict the labels corresponding to the input X. Note that X is a numpy
        array of shape (n_samples, 96, 96, 3) and the output should be a numpy
        array of shape (n_samples,). Therefore, outputs must no be one-hot
        encoded.

        The following is an example of a prediction from the pre-trained model
        loaded in the __init__ method.
        """
        normalized_X = (X / 127.5).astype('float32') - 1  # Normalize to [-1, 1]
        resized_X = np.array([cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in normalized_X])

        preds = self.neural_network.predict(resized_X)
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        return preds
