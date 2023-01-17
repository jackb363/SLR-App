import cv2
import os
from function import *
import mediapipe as mp

root_dir = os.listdir('C:/Users/Jack/Documents/MediaPipe_SmallDataset')

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def extract_nparr_vid():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for folder in root_dir:
            vid_list = get_files(os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset', folder))
            for vid in vid_list:
                # creates dirs for each set of .npy files
                os.mkdir(
                    os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset', folder, str(vid_list.index(vid))))
                # loads current vid to videocapture
                cap = cv2.VideoCapture(os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset', folder, vid))
                frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # to make all vid 30 frames long
                if frames_length > 30:
                    difference = frames_length - 30 / 2


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
                        npy_path = os.path.join('C:/Users/Jack/Documents/MediaPipe_SmallDataset', folder,
                                                str(vid_list.index(vid)), str(frame_num))
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        break
                cap.release()
                cv2.destroyAllWindows()


extract_nparr_vid()
