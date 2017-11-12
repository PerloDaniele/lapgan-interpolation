from pytube import YouTube
import threading
import os
import numpy as np
import constants as c
import utils
from glob import glob
import cv2

GLOBAL_LOCK = threading.Lock()
GLOBAL_CLIP_NUM = 0

def on_complete_download(stream, file_handle):
    global GLOBAL_LOCK
    global GLOBAL_CLIP_NUM
    filename = os.path.join(c.DOWNLOAD_DIR, stream.default_filename)
    clip = full_clips_from_video(filename, 1)[0]
    shape = np.shape(clip)
    H = shape[0]
    W = shape[1]

    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3 * (c.HIST_LEN + 1)])
    for i in range(100):
        crop_x = np.random.choice(W - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(H - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if take_first or utils.clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    GLOBAL_LOCK.acquire()
    clip_num = GLOBAL_CLIP_NUM
    GLOBAL_CLIP_NUM = GLOBAL_CLIP_NUM + 1
    GLOBAL_LOCK.release()

    np.savez_compressed(c.TRAIN_DIR_CLIPS + str(clip_num), cropped_clip)
    if (clip_num + 1) % 100 == 0: 
        print('Processed %d clips' % (clip_num + 1))

def download_youtube_video(url):
    try:
        yt = YouTube(url)
        yt.register_on_complete_callback(on_complete_download)
        stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        stream.download(c.DOWNLOAD_DIR)
    except:
        print("Failed: {}".format(url))    

def full_clips_from_video(video, num_clips):
    stream      = cv2.VideoCapture(video)
    width       = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    clips = np.empty([num_clips,
                    height,#c.FULL_HEIGHT,
                    width,#c.FULL_WIDTH,
                    (3 * (c.HIST_LEN + 1))])

    start_frame = np.random.randint(frame_count - c.HIST_LEN)
    stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for index in range(num_clips):
        _, frame = stream.read()
        #clips[index] = cv2.resize(frame, c.FULL_WIDTH, c.FULL_HEIGHT)
    stream.release()

    return clips

def process_training_data_youtube(url_list_file, num_clips):
    global GLOBAL_CLIP_NUM
    urls = []
    with open(url_list_file) as file:
        for line in file:   
            url = line.split()[0]
            urls.append(url)

    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))
    GLOBAL_CLIP_NUM = num_prev_clips

    for clip_num in range(num_prev_clips, num_clips + num_prev_clips):
        url = np.random.choice(urls)
        download_youtube_video(url)