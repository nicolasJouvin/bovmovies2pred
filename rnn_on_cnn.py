# Forget flake error for now (dev)
# flake8: noqa
from tensorflow.python.ops.script_ops import py_func
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

physical_devices = tf.config.experimental.list_physical_devices("GPU")
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
#         )
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(
#             len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
#         )
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


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
            prediction_times = [27, 54, 94]
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

    def feature_transform(
        self, videos: list, physical_device="/CPU:0", batch_size=32
    ):
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

            # Use the CPU/GPU as you prefer
            # WARNING : if pred_time is large, so is n_frames
            # and the GPU could run out of memory
            with tf.device(physical_device):
                X[i, :, :] = self.feature_extractor.predict(
                    vid_arr_rgb, batch_size=batch_size
                )

        return X

    def fit(
        self,
        videos: list,
        y,
        pred_time: float,
        physical_device="/CPU:0",
        batch_size=32,
    ):

        print("--- Fitting...")
        # transform each videos into a feature set of size (n_frames, self.feature_dimension)

        X = self.feature_transform(videos, physical_device, batch_size)

        seq_model = self.model_dictionnary[int(pred_time)]
        seq_model.fit(
            X,
            self.label_processor(y).numpy(),
            validation_split=0.1,
            epochs=self.epochs,
        )
        print("Done !")

        return None

    def predict(
        self,
        videos: list,
        pred_time: float,
        physical_device="/CPU:0",
        batch_size=32,
    ):
        X = self.feature_transform(videos, physical_device, batch_size)
        y_pred = self.model_dictionnary[int(pred_time)].predict(X)
        return y_pred


if __name__ == "__main__":
    from sklearn.preprocessing import OneHotEncoder

    # we need to convert labels (str) to 1-hot encoding (n, 8)

    pred_time = 54
    videos_train, labels_train = get_train_data()
    n_vid = 40
    clf = VideoClassifier(
        unique_labels=np.unique(labels_train[:n_vid]), epochs=10
    )
    # The labels of the videos are strings. Neural networks do not understand string values,
    # so they must be converted to some numerical form before they are fed to the model.
    # Here we will use the StringLookup layer encode the class labels as integers.

    clf.fit(
        videos=videos_train[:n_vid],
        y=labels_train[:n_vid],
        pred_time=pred_time,
        physical_device="/CPU:0",  # using CPU here
    )

    # Prediction on train & test
    y_pred_train = clf.predict(
        videos_train[:n_vid], pred_time=pred_time, physical_device="/CPU:0"
    )
    # y_pred_test = clf.predict(videos_test[:n_vid], physical_device="/CPU:0")
    print(y_pred_train)
    score = WeightedClassificationError(time_idx=0)

    labels_train = labels_train.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels_train)
    y_true = enc.transform(labels_train)

    print(
        "Error on train = ",
        score(y_true=y_true[:n_vid, :], y_pred=y_pred_train),
    )


    import tensorflow as tf

    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    print(cuda_version)
  