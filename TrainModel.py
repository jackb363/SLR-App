import numpy as np
import os
import util
import pathlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from tensorflow import lite

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/A00275711/Documents/MediaPipe_SmallDataset')
# Actions that we try to detect
actions = np.array(os.listdir(DATA_PATH))
# Videos are going to be 30 frames in length

# logs to monitor training
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# adds labels to each video frame
def label_frame():
    sequences, labels = [], []
    # Label data for training
    label_map = {label: num for num, label in enumerate(actions)}
    number_err = 0
    for action in actions:

        # print(f'labelling videos of sign "{action}"')
        only_dirs = [name for name in os.listdir(os.path.join(DATA_PATH, action)) if
                     os.path.isdir(os.path.join(DATA_PATH, action, name))]
        for sequence in only_dirs:
            # print('processing video', sequence, 'of', str(len(only_dirs) - 1))
            window = []
            list_files = os.listdir(os.path.join(DATA_PATH, action, str(sequence)))

            # sorts files so frames are in order
            sorted_filenames = sorted(list_files, key=util.extract_number)

            for frame_num in sorted_filenames:
                print(frame_num)
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), frame_num))
                window.append(res)
        result = np.where(actions == action)
        print(f'completed labelling of sign {str(result[0])}/{len(actions)}')
        sequences.append(window)
        labels.append(label_map[action])
    print('number of error files :', number_err)
    return sequences, labels


# model structure
def load_model():
    # LSTM model 6 Layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # returns a number of values equal to action types in dataset 'actions.shape[0]'
    model.add(Dense(3, activation='softmax'))
    return model


def train_model(model, X, y):
    # Train/test/val split 80/10/10
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5)

    # stops early if val score has not improved in 5 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # save weights if val score is better than previous
    model_checkpoint = ModelCheckpoint('action.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val),
              callbacks=[early_stopping, model_checkpoint, tb_callback])
    # final weights of trained model saved
    model.save('action.h5')

    # evaluates model after training
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


# saves model structure to json and tflite
def save_model_json_tflite(model):
    # save to json format
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # save to tflite format
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    # saves tflite to current directory
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # call to create and save model struc
    lstm_model = load_model()
    save_model_json_tflite(lstm_model)
    name = input("Type 'train' to train model or press 0 to quit : \n")
    if name == 'train':
        # call to label and group frames of same videos
        sequences, labels = label_frame()
        X = np.array(sequences, dtype=object)
        y = to_categorical(labels).astype(int)

        # call train and pass model struc, frame sequences and associated labels
        train_model(lstm_model, X, y)
    else:
        exit()
