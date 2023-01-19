import cv2
import os

import function
from function import *
import mediapipe as mp

# path to dataset and file list
root_dir = 'C:/Users/A00275711/Documents/MediaPipe_SmallDataset'
root_dir_files = os.listdir('C:/Users/A00275711/Documents/MediaPipe_SmallDataset')

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


# function to extract numpy arrays from video frames
def extract_np_arr():
    # mediapipe is used to extract keypoints from the video
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # iterates over the categories in dataset
        for folder in root_dir_files:
            # gets all videos in a category
            vid_list = get_files(os.path.join(root_dir, folder))
            # iterates over each video in category
            for vid in vid_list:
                # creates dirs for each set of .npy files
                os.mkdir(
                        os.path.join(root_dir, folder, str(vid_list.index(vid))))
                # loads current vid to videocapture
                cap = cv2.VideoCapture(os.path.join(root_dir, folder, vid))
                frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # iterates over each frame in a video
                for frame_num in range(frames_length):
                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    try:
                        image, results = mediapipe_detection(frame, holistic)
                        # Draw landmarks
                        draw_styled_landmarks(image, results)
                        # get keypoints and save to .npy
                        keypoints = extract_keypoints(results)
                        # saves data keypoints to .npy file
                        npy_path = os.path.join(root_dir, folder,
                                                str(vid_list.index(vid)), str(frame_num))
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        break
                cap.release()
                cv2.destroyAllWindows()
                print('category: ', folder)
                # folder loaded containing .npy files
                npy_dir = os.path.join(root_dir, folder, str(vid_list.index(vid)))
                # checks to ensure number of each vid .npy files is 30
                if frames_length < 30:
                    # pads number of .npy files to 30
                    function.pad_npy(npy_dir, frames_length, 30)
                elif frames_length > 30:
                    # cuts number of .npy files to 30
                    function.cut_npy(npy_dir, frames_length, 30)


if __name__ == '__main__':
    extract_np_arr()
