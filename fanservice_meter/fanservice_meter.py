import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os 
from os import path

labels = ("hentai", "shonen")
class fsm_ai:
    #TODO: convert to kwargs
    def __init__(self, img_size = (300, 300), batch_size=8, use_premade = True, make_new = False):
        self.img_size = img_size
        self.batch_size = batch_size
        

        path_prefix = path.dirname(path.abspath(__file__))
        model_path = path.join(path_prefix, 'wholemodel.h5')
        self._model = keras.models.load_model(model_path) if use_premade else None
        try:
            assert not (use_premade and make_new)
        except AssertionError:
            raise ValueError("cannot both use premade model and generate new model")
        if make_new: 
            self._model = self._make_model() 

    def _make_model(self):
        input_shape = (self.img_size[0], self.img_size[1], 3)
        num_classes = 2
        inputs = keras.Input(shape=input_shape)

        data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ])   
        x = data_augmentation(inputs)
        # Entry block
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs) # scale rgb to float
        x = layers.Dropout(0.3)(x) # dropout improve resilience
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256]:
            x = layers.Activation("relu")(x)
            print(x.shape)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)

    def train(self, train_ds, val_ds, epochs = 1):
        train_ds = train_ds.prefetch(buffer_size=32)
        val_ds = val_ds.prefetch(buffer_size=32)
        self._model.compile(
        optimizer=keras.optimizers.Adam(0.001),
            loss = "binary_crossentropy",
            metrics=["accuracy"]
        )
        self._model.fit(
            train_ds, epochs=epochs, validation_data=val_ds,
        )
    
    def save(self):
        name = "model_save"
        type_ = "h5"
        numcopys = 0
        while os.path.exists(name+"({}).".format(numcopys)+type_):
            numcopys += 1
        filename= name+"({}).".format(numcopys)+type_
        self._model.save(filename)

    def load(self, filename):
        try:
            assert filename.split('.')[-1] == 'h5'
        except AssertionError:
            raise TypeError("Attempted to load file that was not of format keras h5")
        self._model = keras.models.load_model(filename)

    def test_img(self, img_in, **kwargs):
        """Accepts params @nullable img_in, img_in can only be None if filename is provided in kwargs"""
        
        img = img_in
        if not img.any():
            img = cv2.imread(kwargs["filename"])
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_array = keras.preprocessing.image.img_to_array(img.astype("uint8"))
        img_array = tf.expand_dims(img_array, 0)
        predictions = self._model.predict(img_array)
        score = predictions[0]
        # hentai is 0, shonen is 1
        label =  labels[0] if 1 - score > 0.5 else labels[1]
        score = 100 * (1 - score) if 1-score > 0.5 else 100 * score
        
        return score, label
