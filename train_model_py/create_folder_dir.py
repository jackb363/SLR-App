import numpy as np
import os
from os.path import exists
import json
import shutil


# gathers all video ids for each word and stores
def vid_id_listings(data):
    id_list = list()
    # loops over each word type in json and adds to list
    for keys in data[0:len(data)]:
        list_of_keys = list()
        video_id_keys = keys['instances']
        # iterates over each video_id and adds to list
        for each_key in video_id_keys:
            list_of_keys.append(each_key.get('video_id'))
        # adds all word types and associated vid ids to dictionary
        id_list.append({keys['gloss']: list_of_keys})
    return id_list


# creates directory for each word type and adds respective data to each
def sort_videos(id_list, path):
    # iterates over each entry of list (key:[values])
    for id in id_list:
        key, value = list(id.items())[0]
        # creates directory for each word in keys list
        os.mkdir(key)
        # iterates over list of video_ids
        for word in value:
            origin_file = path + word + '.mp4'
            dest_dir = key + '/' + word + '.mp4'
            # checks to see if video file exists and sends to new named directory
            file_exists = exists(origin_file)
            if file_exists:
                shutil.move(origin_file, dest_dir)


# opening JSON file
json_file = open('C:/Users/Jack/Downloads/archive/WLASL_v0.3.json')
# stores json as dict
json_data = json.load(json_file)
# directory folders will be in
parentdir = 'C:/Users/Jack/Documents/Project_WordSet'

# dict value to retrieve dataset class names
vid_id_list = vid_id_listings(json_data)
# sort_videos(json_data, 'C:/Users/Jack/Downloads/archive/videos/')

