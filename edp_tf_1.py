
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/kermany2018")
print("Path to dataset files:", path)

# from google.colab import drive
# drive.mount('/content/drive')

# import shutil
# shutil.rmtree('kermany2018')

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import keras
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV3Large



"""## Data Prepration"""

device = tf.test.gpu_device_name()
if device != '/device:GPU:0':
  print('GPU device not found')
else:
  print('Found GPU at: {}'.format(device))

TRAIN_DATA_PATH = path + '/OCT2017 /train'
TEST_DATA_PATH = path + '/OCT2017 /test'
VAL_DATA_PATH = path + '/OCT2017 /val'

training_set = image_dataset_from_directory(
    TRAIN_DATA_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True
)

test_set = image_dataset_from_directory(
    TEST_DATA_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=False
)

val_set = image_dataset_from_directory(
    VAL_DATA_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True
)

"""## Model Building"""

INPUT_SHAPE = (224, 224, 3)

mobnet = MobileNetV3Large(
    input_shape=INPUT_SHAPE,
    alpha=1.0,
    minimalistic=False,
    input_tensor=None,
    include_top=True,
    weights="imagenet",
    pooling=True,
    dropout_rate=0.2,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True
)
mobnet.trainable = True

model = keras.models.Sequential()

model.add(keras.layers.Input(shape=INPUT_SHAPE))
model.add(mobnet)
model.add(keras.layers.Dense(units=4, activation="softmax"))

metrics = ["accuracy", tf.keras.metrics.F1Score()]
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001, ),
    loss="categorical_crossentropy",
    metrics=metrics
)
model.summary()

history = model.fit(training_set, validation_data=val_set, epochs=15)

# from google.colab import drive
# drive.mount('/content/drive')

model.save('/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-trainable.keras')
model.save('/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-trainable.h5')

import pickle
with open("/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-Training_history.pkl", "wb") as file:
  pickle.dump(history.history, file)

with open("/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-Training_history.pkl", "rb") as file:
  history = pickle.load(file)

history.keys()

from google.colab import drive
drive.mount('/content/drive')

epochs = np.arange(1, 16)
plt.plot(epochs, history['loss'], color='red', label='Training Loss')
plt.plot(epochs, history['val_loss'], color='blue', label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.title("Minimization of Loss over Epochs")
plt.savefig('/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-loss-over-epochs.png')
plt.show()

"""### Model Evaluation"""

model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-trainable.keras")

test_loss, test_acc, test_f1_score = model.evaluate(test_set)

true_categories = tf.concat([y for x, y in test_set], axis=0)
y_true = tf.argmax(true_categories, axis=1)

predicted_categories = model.predict(test_set)

y_pred = tf.argmax(predicted_categories, axis=1)
y_pred

"""### Classification Report"""

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

"""## Confusion Matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
cm

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, annot_kws={"size": 8})
plt.xlabel("Predicted Class", fontsize=10)
plt.ylabel("Actual Class", fontsize=10)
plt.title("Human Eye Disease Prediction Confusion Matrix", fontsize=12)
plt.savefig("HeatMap.png")
plt.savefig('/content/drive/MyDrive/Colab Notebooks/CV/Eye Disease Prediction/models/EDP-tf-1-heat-map.png')



