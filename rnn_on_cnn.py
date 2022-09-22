# Forget flake error for now (dev)
# flake8: noqa
from problem import (
    get_train_data,
    get_test_data,
    WeightedClassificationError,
    VideoReader,
)
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


class VideoClassifier:
    def __init__(
        self,
        feature_extractor_name="inceptionV3",
        augment=False,
        batch_size=16,
        epochs=2,
        begin_time=24.0,
        unique_labels=None,
        prediction_times=None,
    ):

        self.augment = augment
        self.fe_name = feature_extractor_name
        self.build_feature_extractor()  # Creates self.feature_extractor and self.feature_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.begin_time = begin_time
        if unique_labels is None:
            unique_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.label_processor = keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=unique_labels
        )
        if prediction_times is None:
            prediction_times = [27, 30]
        self.model_dictionnary = {}
        for pred_time in prediction_times:
            self.model_dictionnary[pred_time] = self.get_sequence_model(
                pred_time
            )

    def build_feature_extractor(self):
        if self.fe_name == "inceptionV3":
            feature_extractor = keras.applications.InceptionV3(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(250, 250, 3),
            )
            preprocess_input = keras.applications.inception_v3.preprocess_input

            inputs = keras.Input((250, 250, 3))
            preprocessed = preprocess_input(inputs)

            outputs = feature_extractor(preprocessed)
            self.feature_extractor = keras.Model(
                inputs, outputs, name="feature_extractor"
            )
            self.feature_dimension = 2048

            return None

    def get_sequence_model(self, pred_time):
        class_vocab = self.label_processor.get_vocabulary()
        n_frames = int((pred_time - self.begin_time) * 4 + 1)
        frame_features_input = keras.Input((n_frames, self.feature_dimension))
        # mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
        # NO mask for the moment as we don't want to complicate stuff
        x = keras.layers.GRU(16, return_sequences=True)(frame_features_input)
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(8, activation="relu")(x)
        output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

        rnn_model = keras.Model([frame_features_input], output)

        rnn_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return rnn_model

    def fit(self, videos: list, y, pred_time: float):
        # transform each videos into a feature set of size (n_frames, self.feature_dimension)

        n_vids = len(videos)
        n_frames = int((pred_time - self.begin_time) * 4 + 1)
        X = np.zeros((n_vids, n_frames, self.feature_dimension))

        for i, vid in enumerate(videos):
            vid_arr = vid.read_sequence(
                begin_time=self.begin_time, end_time=pred_time
            )
            # convert gray to RGB channel by copying 3 times
            vid_arr_rgb = np.array([vid_arr] * 3)
            vid_arr_rgb = np.rollaxis(vid_arr_rgb, 0, vid_arr_rgb.ndim)

            X[i, :, :] = self.feature_extractor.predict(
                vid_arr_rgb, batch_size=33
            )
        seq_model = self.model_dictionnary[int(pred_time)]
        seq_model.fit(
            X,
            self.label_processor(y).numpy(),
            validation_split=0.1,
            epochs=self.epochs,
        )

        return None

    def predict(self, videos: list, pred_time: float):

        return None


if __name__ == "__main__":
    pred_time = 30
    videos_train, labels_train = get_train_data()
    n_vid = 10
    clf = VideoClassifier(unique_labels=np.unique(labels_train[:n_vid]))
    # The labels of the videos are strings. Neural networks do not understand string values,
    # so they must be converted to some numerical form before they are fed to the model.
    # Here we will use the StringLookup layer encode the class labels as integers.

    train_preds = clf.fit(
        videos=videos_train[:n_vid],
        y=labels_train[:n_vid],
        pred_time=pred_time,
    )
