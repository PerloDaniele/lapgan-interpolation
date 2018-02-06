import numpy as np
import sys
import os
from glob import glob
    

def main():
    path = [sys.argv[1]]
    if os.path.isdir(path[0]):
        clips = glob(os.path.join(path[0], '*.npz'))
    
    #meanfile as second argument, use it for prepare the TestSet
    mean_calc = True;
    if len(sys.argv)>=3:
        mean_file = [sys.argv[2]] 
        mean = clip = np.load(mean_file[0])['arr_0']
        mean_calc=False
        
        
    
    clip = np.load(clips[0])['arr_0']
    shape = np.shape(clip)
    h = shape[0]
    w = shape[1]
    num_frames = int(shape[2] / 3)
    #mean = np.zeros([h,w, num_frames ,3])
    dest = os.path.join(path[0], 'zero_c')
    if not os.path.exists(dest):
        os.makedirs(dest)
            
    if mean_calc:
        mean = np.zeros([3,])
        num_img = len(clips) * num_frames
    
        for clip_file in clips:
            clip = np.load(clip_file)['arr_0']
        
            #3 channel for each pixel per each clip image
            frames = np.reshape(clip, (-1, 3))
            mean += np.sum(frames, axis=0)
            #old Code
            #frames = np.reshape(clip, (h, w, num_frames, 3))
            #mean += np.sum(frames, axis=2, keepdims=True)
            
        mean /= (num_img * h * w)
        #mean = np.reshape(mean, (h, w, num_frames * 3))    
        metadest = os.path.join(dest,'metadata')
        if not os.path.exists(metadest):
            os.makedirs(metadest)
        np.savez_compressed(os.path.join(metadest,'mean_' + str(len(clips))), mean)

    #useless
    '''
    #inv_var = 1/np.std(frames, axis=2, keepdims=True)
    var = np.zeros([h,w, num_frames * 3])
    for clip_file in clips:
        clip = np.load(clip_file)['arr_0']
        #frames = np.reshape(clip, (h, w, num_frames, 3))
        var += np.sum(np.square(clip-mean), axis=2, keepdims=True)
    var /= num_img
    inv_std = 1/np.sqrt(var) #1/np.std(frames, axis=2, keepdims=True)
    
    np.savez_compressed(os.path.join(metadest,'inv_std_' + str(len(clips))), inv_std)
    '''
    
    for clip_file in clips:
        clip = np.load(clip_file)['arr_0']
        #clip = (clip-mean) * inv_std
        #normalize, (clip -mean ) is [-2,2]
        clip = np.reshape(clip, (h, w, num_frames, 3))
        #clipping, only outsider pixels are involved
        clip = np.clip(clip - mean, -1, 1)
        clip = np.reshape(clip, (h, w, num_frames * 3))
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