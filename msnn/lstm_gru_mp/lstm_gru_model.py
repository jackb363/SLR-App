from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Masking, GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from focal_loss_func import focal_loss


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Jack/Documents/refined_dataset')
# Actions that we try to detect
actions = np.array(os.listdir(DATA_PATH))

# logs to monitor training
log_dir = os.path.join('Logs')
# invoke on cmd line with 'tensorboard --logdir=.' from 'Logs' dir
tb_callback = TensorBoard(log_dir=log_dir)


def build_model(max_seq_len):
    # LSTM model 6 Layers
    model = Sequential()
    # removes zeroed arrays from sequences / allows for var length sequences
    model.add(Masking(mask_value=0.0, input_shape=(max_seq_len, 225)))
    model.add(LSTM(128, dropout=0.3, return_sequences=True, activation='relu', kernel_regularizer=l2()))
    model.add(GRU(128, return_sequences=False, dropout=0.3, kernel_regularizer=l2()))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_lstmgru_model(model, X, y):
    model.compile(loss=focal_loss(gamma=2.0, alpha=0.25), optimizer=Adam(learning_rate=0.001), metrics=['categorical_accuracy'])
    model_checkpoint = ModelCheckpoint('../res/lstm_gru.h5', save_best_only=True, monitor='val_loss', mode='min')
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    # Train the model on each fold of the train set
    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Fit the model on the train fold
        
        model.fit(X_train_fold, y_train_fold, batch_size=16, epochs=20, validation_data=(X_val_fold, y_val_fold), callbacks=[model_checkpoint, tb_callback])
    # Evaluate the final model on the test set
    model.save('../res/lstm_gru.h5')
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


# adds labels to each video frame
def label_frame():
    sequences, labels = [], []
    # Label data for training
    label_map = {label: num for num, label in enumerate(actions)}
    number_err = 0
    for action in actions:

        print(f'labelling videos of sign "{action}"')
        only_dirs = [name for name in os.listdir(os.path.join(DATA_PATH, action)) if
                     os.path.isdir(os.path.join(DATA_PATH, action, name))]
        for npy_dir in only_dirs:
            # print('processing video', sequence, 'of', str(len(only_dirs) - 1))
            window = []
            list_files = os.listdir(os.path.join(DATA_PATH, action, str(npy_dir)))

            # sorts files so frames are in order
            sorted_files = sorted(list_files, key=extract_number)

            for frame_num in sorted_files:
                # print(frame_num)
                res = np.load(os.path.join(DATA_PATH, action, str(npy_dir), frame_num))
                window.append(res)
            result = np.where(actions == action)
            print(f'completed labelling of sign {str(result[0])}/{len(actions)}')
            sequences.append(window)
            labels.append(label_map[action])
    print('number of error files :', number_err)
    return sequences, labels


# extracts numeric part of filename
def extract_number(file_name):
    return int(os.path.splitext(file_name)[0])


if __name__ == '__main__':
    # call to label and group frames of same videos
    sequences, labels = label_frame()
    max_len = max(len(seq) for seq in sequences)
    # print(max_len)
    X = pad_sequences(sequences, value=0.0, maxlen=221, padding='post', dtype='float32')
    y = to_categorical(labels).astype(int)
    # call to create and save model struc
    lstm_gru_model = build_model(max_len)
    # call train and pass model struc, frame sequences and associated labels
    train_lstmgru_model(lstm_gru_model, X, y)
    # save_model_tflite('../res/action.h5')
