import os
import random
import cv2
import shutil

in_dir = 'C:/Users/Jack/Documents/MediaPipe_SmallDataset'
categories = os.listdir(in_dir)


def get_anno_file():
    count = 0
    for folder in categories:
        vids = os.listdir(os.path.join(in_dir, folder))
        # Shuffle the list of filenames randomly
        random.shuffle(vids)

        # Calculate the split index based on the 80/20 split ratio
        split_idx = int(0.9 * len(vids))
        # Split the list of filenames into two separate lists
        train_videos = vids[:split_idx]
        test_videos = vids[split_idx:]
        names = []
        for video in train_videos:
            cap = cv2.VideoCapture(os.path.join(in_dir, folder, video))
            # Get the total number of frames in the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            file_entry = video + ' ' + str(frame_count) + ' ' + str(count)
            names.append(file_entry)
            cap.release()
        with open('train.txt', 'a') as f:
            for line in names:
                f.write(line)
                f.write('\n')
        names = []
        for video in test_videos:
            cap = cv2.VideoCapture(os.path.join(in_dir, folder, video))
            # Get the total number of frames in the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Release the video capture object
            file_entry = video + ' ' + str(frame_count) + ' ' + str(count)
            names.append(file_entry)
            cap.release()
        with open('val.txt', 'a') as f:
            for line in names:
                f.write(line)
                f.write('\n')
        count += 1


def move_train_val(anno_file, out_dir, type_file):
    file = open(anno_file, 'r')
    data = file.read()
    split_data = data.split('\n')

    for vid in split_data:
        info = vid.split(' ')
        try:
            if not os.path.exists(os.path.join(out_dir, type_file, info[2])):
                os.mkdir(os.path.join(out_dir, type_file, info[2]))
            shutil.copy(os.path.join(in_dir, categories[int(info[2])], info[0]), os.path.join(out_dir, type_file, info[2], info[0]))
        except Exception as e:
            print('error on empty entry')



if __name__ == '__main__' :
    move_train_val('val.txt', 'C:/Users/Jack/Documents/train_val_split', 'val')
    move_train_val('train.txt', 'C:/Users/Jack/Documents/train_val_split', 'train')

