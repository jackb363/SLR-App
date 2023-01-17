import numpy as np
import os
import fnmatch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset')
print(os.listdir(DATA_PATH))
# Actions that we try to detect
actions = np.array(os.listdir(DATA_PATH))
# Videos are going to be 30 frames in length

# logs to monitor training
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

sequences, labels = [], []


# adds labels to each video frame
def label_frame():
    # Label data for training
    label_map = {label: num for num, label in enumerate(actions)}
    for action in actions:

        print(f'labelling videos of sign "{action}"')
        only_dirs = [name for name in os.listdir(os.path.join(DATA_PATH, action)) if
                     os.path.isdir(os.path.join(DATA_PATH, action, name))]
        for sequence in only_dirs:
            print('processing video ', sequence, 'of', str(len(only_dirs) - 1))
            # gets number of .npy files
            count = len(fnmatch.filter(os.listdir(os.path.join(DATA_PATH, action, sequence)), '*.*'))
            sequence_length = count
            window = []

            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
        result = np.where(actions == action)
        print(f'completed labelling of sign {str(result[0])}/{len(actions)}')
        sequences.append(window)
        labels.append(label_map[action])


label_frame()

# Train/test split
X = np.array(sequences, dtype=object)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03)

print(X.shape)


def load_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # returns a number of values equal to action types in dataset
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_model(model):
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.save('action.h5')

# call load and train

# lstm_model = load_model()
# train_model(lstm_model)
