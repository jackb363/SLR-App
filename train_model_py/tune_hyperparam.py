from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from kerastuner import RandomSearch
import train_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset')
# Actions that we try to detect
actions = np.array(os.listdir(DATA_PATH))


# Define the model-building function with hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(221, 225)))
    model.add(
        LSTM(units=hp.Int('unit_1', min_value=32, max_value=256, step=32), return_sequences=True, activation='relu'))
    model.add(
        LSTM(units=hp.Int('unit_2', min_value=32, max_value=256, step=32), return_sequences=True, activation='relu'))
    model.add(
        LSTM(units=hp.Int('unit_3', min_value=32, max_value=256, step=32), return_sequences=False, activation='relu'))
    model.add(Dense(units=hp.Int('unit_4', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dense(units=hp.Int('unit_5', min_value=16, max_value=128, step=16), activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='my_dir',
    project_name='helloworld')

if __name__ == '__main__':
    # call to label and group frames of same videos
    sequences, labels = train_model.label_frame()
    X = pad_sequences(sequences, value=0.0, maxlen=221, padding='post', dtype='float32')
    y = to_categorical(labels).astype(int)

    # Instantiate the tuner and perform hypertuning
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5)
    tuner.search_space_summary()

    tuner.search(X_train, y_train,
                 epochs=10,
                 validation_data=(X_val, y_val))

    # Print the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps}")