import os
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import video

# Set up model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(35, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Set up data directories
data_dir = 'C:/Users/Jack/Documents/MediaPipe_SmallDataset'

# set up the generator
train_datagen = video.VideoFrameGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(data_dir),
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(data_dir),
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# invoke on cmd line with 'tensorboard --logdir=.' from 'Logs' dir
tb_callback = TensorBoard(log_dir='/Logs')
# save weights if val score is better than previous
model_checkpoint = ModelCheckpoint('../res/i3d_wlasl.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[model_checkpoint, tb_callback])
model.save('../res/i3d_wlasl.h5')
