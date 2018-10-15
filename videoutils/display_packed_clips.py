import numpy as np
import cv2
import sys
import os
from glob import glob

def denormalize_frames(frames):
    new_frames = frames + 1
    new_frames *= 127.5 #(255 / 2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def main():
    path = [sys.argv[1]]
    if os.path.isdir(path[0]):
        path = glob(os.path.join(path[0], '*.npz'))
        
    #usage --zero_c "pathtotrainingsetmean/mean.npz"
    mean = np.array([])
    if len(sys.argv)>=4 and sys.argv[2] == '--mean' :
        mean = np.load(sys.argv[3])['arr_0']
    
    while True:
        clip_file = np.random.choice(path)
        clip = np.load(clip_file)['arr_0']
        #print(str(np.min(clip)) + '\t' + str(np.max(clip)))
        shape = np.shape(clip)
        h = shape[0]
        w = shape[1]
        num_frames = int(shape[2] / 3)
        frames = np.reshape(clip, (h, w, num_frames, 3))
        #adding mean
        if mean.shape[0] != 0:
            frames = np.clip((frames + mean),-1,1)
            
        for frame_index in range(num_frames):
            frame = denormalize_frames(frames[:,:,frame_index,:])
            cv2.imshow(str(frame_index), frame[:,:,[2,1,0]])    
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:
            break
    
if __name__ == '__main__':
    main()