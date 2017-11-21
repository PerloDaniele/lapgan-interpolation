import numpy as np
import cv2
import sys

def denormalize_frames(frames):
    new_frames = frames + 1
    new_frames *= (255 / 2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def main():
    path = sys.argv[1]
    clip = np.load(path)['arr_0']
    shape = np.shape(clip)
    h = shape[0]
    w = shape[1]
    num_frames = int(shape[2] / 3)
    frames = np.reshape(clip, (h, w, num_frames, 3))
    for frame_index in range(num_frames):
        frame = denormalize_frames(frames[:,:,frame_index,:])
        cv2.imshow(str(frame_index), frame)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()