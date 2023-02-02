import cv2
import numpy as np
import mediapipe as mp
import os
from multiprocessing import Process
from natsort import natsorted

# Holistic model
mp_holistic = mp.solutions.holistic
# Drawing utilities
mp_drawing = mp.solutions.drawing_utils


# function to call mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


# function uses mediapipe to extract keypoints from frames
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


# function to superimpose keypoints over frame
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


# function that gets only files ignores folders
def get_files(path):
    file_names = list()
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            file_names.append(file)
    return file_names


# function to pad .npy files to a specified number
def pad_npy(npy_dir, current_files, new_file_max):
    # create numpy zeros array of same dimension as other files
    zeros_array = np.array([np.zeros(33*4), np.zeros(468*4), np.zeros(21*3), np.zeros(21*3)])
    # loop to create new .npy files and copy them to a directory
    for copy_frame in range(current_files, new_file_max):
        padded_file = zeros_array
        npy_path = os.path.join(npy_dir, str(copy_frame))
        np.save(npy_path, padded_file)


# function takes the middle number of .npy files specified
def cut_npy(npy_dir, current_file_num, new_file_num):
    file_list = natsorted(os.listdir(npy_dir))


# removes the files at the start and end of directory
def remove_files(dir_path, start_file, end_file):
    for filename in range(start_file, end_file):
        os.remove(os.path.join(dir_path, str(filename) + '.npy'))


# runs tasks in parallel
def run_parallel(*functions):
    processes = []
    for function in functions:
        proc = Process(target=function)
        proc.start()
        processes.append(proc)
    for process in processes:
        process.join()


# delete all files in a specified directory
def delete_all_files_in_dir(dir_path):
    current_dir = os.path.split(dir_path)
    print('Deleting Files For Directory', current_dir[1])
    for file in os.scandir(dir_path):
        os.remove(file.path)


# extracts numeric part of filename
def extract_number(file_name):
    return int(os.path.splitext(file_name)[0])
