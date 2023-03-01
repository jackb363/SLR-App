import os
import random
import shutil

data_dir = 'C:/Users/Jack/Documents/MediaPipe_SmallDataset'
train_dir = 'C:/Users/Jack/Documents/Train_Test_Split/train'
val_dir = 'C:/Users/Jack/Documents/Train_Test_Split/validation'
val_split_ratio = 0.2

# create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# loop over the video categories and split them into train and validation sets
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    train_category_path = os.path.join(train_dir, category)
    val_category_path = os.path.join(val_dir, category)
    print(os.listdir(category_path))
    # create train and validation category directories
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(val_category_path, exist_ok=True)

    # get list of video files in category directory
    videos = os.listdir(category_path)

    # shuffle the list of video files
    random.shuffle(videos)

    # split the list of video files into train and validation sets
    val_videos = videos[:int(len(videos) * val_split_ratio)]
    train_videos = videos[int(len(videos) * val_split_ratio):]

    # move video files to train or validation category directory
    for video in train_videos:
        src = os.path.join(category_path, video)
        dst = os.path.join(train_category_path, video)
        shutil.move(src, dst)

    for video in val_videos:
        src = os.path.join(category_path, video)
        dst = os.path.join(val_category_path, video)
        shutil.move(src, dst)