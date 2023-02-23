import numpy as np
import os
import util
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from tensorflow import lite

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Jack/Documents/WLASL_Refined_Dataset')
# Actions that we try to detect
actions = np.array(os.listdir(DATA_PATH))
# Videos are going to be 30 frames in length

# logs to monitor training
log_dir = os.path.join('Logs')
# invoke on cmd line with 'tensorboard --logdir=.' from 'Logs' dir
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
def build_model(max_seq_len):
    # LSTM model 6 Layers
    model = Sequential()
    # removes zeroed arrays from sequences / allows for var length sequences
    model.add(Masking(mask_value=0.0, input_shape=(max_seq_len, 1662)))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # returns a number of values equal to action types in dataset
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_model(model, X, y):
    # Train/test/val split 80/10/10
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5)
    print(X_train.shape,'\n', X_test.shape, '\n', X_val.shape)
    # stops early if val score has not improved in 5 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # save weights if val score is better than previous
    model_checkpoint = ModelCheckpoint('../res/action.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val),
              callbacks=[early_stopping, model_checkpoint, tb_callback])

    # final weights and structure of trained model saved
    model.save('../res/action.h5')

    # evaluates model after training
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


# saves model structure to json
def save_model_struc(model):
    model_json = model.to_json()
    with open('../res/model.json', 'w') as json_file:
        json_file.write(model_json)


# saves model struc and weights to tflite file
def save_model_tflite(h5_file):
    model = load_model(h5_file)
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    open("../res/model.json", "wb").write(tflite_model)


if __name__ == '__main__':
    # call to label and group frames of same videos
    sequences, labels = label_frame()
    max_len = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')
    y = to_categorical(labels).astype(int)
    # call to create and save model struc
    lstm_model = build_model(max_len)
    save_model_struc(lstm_model)

    # call train and pass model struc, frame sequences and associated labels
    train_model(lstm_model, X, y)
    #save_model_tflite('../res/action.h5')
