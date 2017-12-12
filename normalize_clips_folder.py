import numpy as np
import sys
import os
from glob import glob

def main():
    path = [sys.argv[1]]
    if os.path.isdir(path[0]):
        clips = glob(os.path.join(path[0], '*.npz'))
    
    clip = np.load(clips[0])['arr_0']
    shape = np.shape(clip)
    h = shape[0]
    w = shape[1]
    num_frames = int(shape[2] / 3)
    mean = np.zeros([h,w, num_frames ,3])
    num_img = len(clips) * num_frames
    
    for clip_file in clips:
        clip = np.load(clip_file)['arr_0']
        frames = np.reshape(clip, (h, w, num_frames, 3))
        mean += np.sum(frames, axis=2, keepdims=True)

    mean /= num_img
    mean = np.reshape(mean, (h, w, num_frames * 3))
    dest = os.path.join(path[0], 'zero_c')
    np.savez_compressed(os.path.join(dest,'mean','mean_' + str(len(clips))), mean)
    
    #inv_var = 1/np.std(frames, axis=2, keepdims=True)
    var = np.zeros([h,w, num_frames * 3])
    for clip_file in clips:
        clip = np.load(clip_file)['arr_0']
        #frames = np.reshape(clip, (h, w, num_frames, 3))
        var += np.sum(np.square(clip-mean), axis=2, keepdims=True)
    var /= num_img
    inv_var = 1/var #1/np.std(frames, axis=2, keepdims=True)
    
    np.savez_compressed(os.path.join(dest,'mean','inv_var_' + str(len(clips))), inv_var)
    
    for clip_file in clips:
        clip = np.load(clip_file)['arr_0']
        clip = (clip-mean) * inv_var
        #prova
        clip = (((2.5 + clip)*2)/5)-1
        np.savez_compressed(os.path.join(dest, os.path.split(clip_file)[1]), clip)
    
    '''
        for frame_index in range(num_frames):
            frame = denormalize_frames(frames[:,:,frame_index,:])
            cv2.imshow(str(frame_index), frame)    
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:
            break
    '''
if __name__ == '__main__':
    main()