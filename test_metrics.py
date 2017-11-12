import sys
import os
import numpy as np
from glob import glob
from sklearn.metrics import mean_absolute_error
from optical_flow import interpolate
import cv2

def main():
    directory = sys.argv[1]
    output_directory = directory#'../Interpolation/'
    image_paths = sorted(glob(os.path.join(directory, '*')))
    shape = np.shape(cv2.imread(image_paths[0]))
    height = shape[0]
    width = shape[1]

    for i in range(1, len(image_paths) - 1):
        img0 = cv2.imread(image_paths[i - 1])
        img1 = cv2.imread(image_paths[i + 1])
        img_gt = cv2.imread(image_paths[i])
        img_pred = interpolate(img0, img1, 0.5)

        MAE = mean_absolute_error(img_gt, img_pred)
        print(MAE)

        split_name  = os.path.splitext(os.path.basename(image_paths[i]))
        name = split_name[0] + "_pred" + split_name[1]
        cv2.imwrite(os.path.join(output_directory, name), img_pred)

def mean_absolute_error(A, B):
    fA = A.astype(np.float)
    fB = B.astype(np.float)
    mae = np.sum(np.absolute(fB - fA))
    mae /= np.prod(np.shape(fA))
    return mae

if __name__ == '__main__':
    main()
