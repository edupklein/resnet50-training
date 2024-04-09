#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
import datetime
from tensorflow.keras.applications import ResNet50

from tensorflow.keras import layers

img_width  = 1024
img_height = 1024
batch_size = 3
data_dir   = "/content/gdrive/My Drive/Topicos/train_dataset/imgs/"

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='categorical'
  )

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='categorical'
  )

classes = train_data.class_names
print(classes)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomContrast(0.2,0.2)
    ]
)

resnet = ResNet50(
    input_shape=(img_width,img_height,3),
    include_top=False,
    weights="imagenet",
    pooling=None,
    classes=len(classes),
)

# resnet.summary()


model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),  
  resnet,
  layers.Conv2D(len(classes),1,2),
  layers.GlobalMaxPooling2D(),
  layers.Activation("softmax")
])

model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1, nesterov=True),
  loss=tf.keras.losses.categorical_crossentropy,
  metrics=['accuracy'])

model.build(input_shape=(batch_size,img_width,img_height,3))
model.summary()

model.fit(
  train_data,
  validation_data=val_data,
  epochs=10,
  shuffle=True
)
