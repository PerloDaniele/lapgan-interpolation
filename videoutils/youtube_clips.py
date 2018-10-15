from pytube import YouTube
import threading
import os
import numpy as np
import getopt

import sys
sys.path.append(os.path.abspath('../'))

from model import constants as c
from model import utils
from glob import glob
import cv2
import time


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

def download_youtube_video(url, on_complete_callback = None):
    try:
        yt = YouTube(url)
        yt.register_on_complete_callback(on_complete_callback)
        stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        stream.download(c.DOWNLOAD_DIR)
        print("Downloading: {}".format(url)) 
    except:
        print("Failed: {}".format(url))    

def get_full_clips(data_dir, num_clips):
    """
    Loads a batch of random clips from the unprocessed train or test data.
    NOTE: the target frame was moved to be the last one. 
    [<HIST_LEN/2 before frames> | <HIST_LEN/2 after  frames> | <frame to be interpolated>]

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    video_list = glob(os.path.join(data_dir, '*'))
    while True:
        video = np.random.choice(video_list)
        ok, clips_rgb = full_clips_from_video(video, num_clips)
        if ok: 
            break
    
    shape = np.shape(clips_rgb)
    clips = np.empty([num_clips,
                      shape[1],
                      shape[2],
                      (3 * (c.HIST_LEN + 1))])
    
    middle = int(c.HIST_LEN / 2)
    frame_indices = list(i for j in (range(middle), [c.HIST_LEN], range(middle, c.HIST_LEN)) for i in j) # in cosa mi sta trasformando python?
    for clip_num in range(num_clips):
        for frame_index_src in range(c.HIST_LEN + 1):
            frame_index_dest = frame_indices[frame_index_src]
            clips[clip_num, :, :, frame_index_dest * 3:(frame_index_dest + 1) * 3] = utils.normalize_frames(clips_rgb[clip_num, :, :, frame_index_src * 3:(frame_index_src + 1) * 3])

    assert(np.max(clips) <= 1.0)
    assert(np.min(clips) >= -1.0)
    return clips

def full_clips_from_video(video, num_clips):
     #GBR Frames
    stream      = cv2.VideoCapture(video)
    #stream.set(cv2.CAP_PROP_CONVERT_RGB,True)
    width       = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate  = int(stream.get(cv2.CAP_PROP_FPS))
    frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT) - frame_rate)
    if frame_count < c.HIST_LEN + 1 or width == 0 or height == 0:
        print('Errore lettura file: {}'.format(video))
        return (False, None)
    
    clips = np.empty([num_clips,
                    height,#c.FULL_HEIGHT,
                    width,#c.FULL_WIDTH,
                    (3 * (c.HIST_LEN + 1))])

    start_frames = np.random.randint(0, frame_count - c.HIST_LEN, num_clips)
    #print(str(start_frames) + ' / ' + str(frame_count))
    for clip_index in range(num_clips):
        stream.set(cv2.CAP_PROP_POS_FRAMES, start_frames[clip_index])
        for frame_index in range(c.HIST_LEN + 1):
            ret, frame = stream.read()
            if not ret:
                print('Errore lettura frame!')
                stream.release()
                return full_clips_from_video(video, num_clips)
            clips[clip_index, :, :, 3 * frame_index : 3 * (frame_index + 1)] = frame
            
    stream.release()
    return (True, clips)

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
        download_youtube_video(url, on_complete_download)
        time.sleep(1)
        
def download_list_youtube(url_list_file):
    urls = []
    with open(url_list_file) as file:
        for line in file:   
            url = line.split()[0]
            download_youtube_video(url)
            im_not_a_robot_i_swear_look_im_so_random = np.random.rand() * 2;
            time.sleep(1 + im_not_a_robot_i_swear_look_im_so_random)
        
if __name__ == "__main__":

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'v:d', ['video=', 'dir='])
    except getopt.GetoptError:
        print('Invalid parameters. use -v \'youtubeUrl\' -d \'pathDir\'')
        sys.exit(2)

    video = 'https://www.youtube.com/watch?v=7N3ERfi6WHM'
    for opt, arg in opts:
        if opt in ('-v', '--video'):
            video = arg
        if opt in ('-d', '--dir'):
            c.DOWNLOAD_DIR = arg

    download_youtube_video(video)

