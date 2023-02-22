import numpy as np
import os
import util
import pathlib
from tensorflow import data
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from tensorflow import lite

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset')
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
        for npy_dir in only_dirs:
            # print('processing video', sequence, 'of', str(len(only_dirs) - 1))
            window = []
            list_files = os.listdir(os.path.join(DATA_PATH, action, str(npy_dir)))

            # sorts files so frames are in order
            sorted_files = sorted(list_files, key=util.extract_number)

            for frame_num in sorted_files:
                #print(frame_num)
                res = np.load(os.path.join(DATA_PATH, action, str(npy_dir), frame_num))
                window.append(res)
            result = np.where(actions == action)
            print(f'completed labelling of sign {str(result[0])}/{len(actions)}')
            sequences.append(window)
            labels.append(label_map[action])
    print('number of error files :', number_err)
    return sequences, labels


# model structure
def build_model():
    # LSTM model 6 Layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # returns a number of values equal to action types in dataset
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_model(model, X, y):
    # creates dataset with varied sequence length
    dataset = data.Dataset.from_tensor_slices((X, y))
    # Define the batch size
    batch_size = 32

    # Shuffle and batch the dataset
    dataset.shuffle(buffer_size=len(X))
    dataset.batch(batch_size)
    dataset.prefetch(buffer_size=data.experimental.AUTOTUNE)

    # Train/test/val split 80/10/10
    dataset_train = dataset.take(int(len(sequences) * 0.8))
    dataset_test = dataset.skip(int(len(sequences) * 0.8))
    dataset_val = dataset_test.take(len(dataset_test) // 2)
    dataset_test = dataset_test.skip(len(dataset_val))

    # stops early if val score has not improved in 5 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # save weights if val score is better than previous
    model_checkpoint = ModelCheckpoint('res/action.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(dataset_train, epochs=2000, validation_data=dataset_val,
              callbacks=[early_stopping, model_checkpoint, tb_callback])

    # final weights and structure of trained model saved
    model.save('action.h5')

    # evaluates model after training
    loss, accuracy = model.evaluate(dataset_test, verbose=0)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


# saves model structure to json
def save_model_struc(model):
    model_json = model.to_json()
    with open('res/model.json', 'w') as json_file:
        json_file.write(model_json)


# saves model struc and weights to tflite file
def save_model_tflite(h5_file):
    model = load_model(h5_file)
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    open("res/model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    # call to create and save model struc
    lstm_model = build_model()
    #save_model_struc(lstm_model)

    # call to label and group frames of same videos
    sequences, labels = label_frame()
    X = np.array(np.concatenate(sequences, axis=0))
    y = to_categorical(labels).astype(int)

    # call train and pass model struc, frame sequences and associated labels
    train_model(lstm_model, X, y)
    #save_model_tflite('res/action.h5')
