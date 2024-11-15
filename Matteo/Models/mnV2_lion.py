from keras import layers as tfkl
from keras import applications as tfka
import keras as tfk

def build_model(input_shape, output_shape, learning_rate, fine_tune_at=20):
    weight_decay = 1e-4
    # Load MobileNetV2 as the base model, which is smaller and efficient
    base_model = tfka.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet",
        alpha=0.75  # Control model width for lightweight (0.35, 0.5, 0.75, or 1.0)
    )
    # Freeze the entire base model initially
    base_model.trainable = False

    # Define the custom head for your dataset
    inputs = tfkl.Input(shape=input_shape, name="Input")
    x = base_model(inputs, training=False)  # Use the base model
    x = tfkl.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tfkl.Dropout(0.2, name="dropout")(x)  # Reduce dropout rate for lighter regularization
    outputs = tfkl.Dense(units=output_shape, dtype='float32', activation="softmax", name="softmax")(x)

    model = tfk.Model(inputs, outputs, name="MobileNetV2_Light")

    # Compile the model with the initial learning rate
    loss = tfk.losses.CategoricalCrossentropy()
    optimizer = tfk.optimizers.Lion(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=weight_decay
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Fine-tuning: Unfreeze part of the base model if specified
    if fine_tune_at is not None:
        base_model.trainable = True  # Unfreeze the whole base model
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False  # Keep all layers except top fine_tune_at frozen

        # Recompile the model with a lower learning rate for fine-tuning
        fine_tune_lr = learning_rate * 0.1
        optimizer = tfk.optimizers.Lion(
            learning_rate=fine_tune_lr,
            beta_1=0.9,
            beta_2=0.99,
            weight_decay=weight_decay
        )
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model
