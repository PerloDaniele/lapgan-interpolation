from avg_runner import AVGRunner
import constants as c
import getopt
import sys
import os
import numpy as np
import cv2
from glob import glob
from sklearn.metrics import mean_absolute_error

def main():
	try:
		opts, _ = getopt.getopt(sys.argv[1:], 'l:d', ['load_path=', 'directory='])
	except getopt.GetoptError:
		print('Invalid parameters.')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-l', '--load_path'):
			load_path = arg
		if opt in ('-d', '--directory'):
			directory = arg
	assert os.path.exists(directory)


	dirs = glob(os.path.join(directory, '*'))
	for dir in dirs:
		print(dir)
		image_paths = sorted(glob(os.path.join(dir, '*')))
		shape = np.shape(cv2.imread(image_paths[0]))
		c.FULL_HEIGHT = shape[0]
		c.FULL_WIDTH  = shape[1]
		images = np.empty([len(image_paths), c.FULL_HEIGHT, c.FULL_WIDTH, 3])
		for i, image in enumerate(image_paths):
			images[i, :, :, :] = cv2.imread(image)
	
		runner = AVGRunner(0, load_path)
		input_frames = np.empty([1, c.FULL_HEIGHT, c.FULL_WIDTH, 3 * c.HIST_LEN], dtype=np.uint8)
		for i in range(2, len(images) - 2):
			input_frames[0, :, :, :3]	= images[i - 2]
			input_frames[0, :, :, 3:6]	= images[i - 1]
			input_frames[0, :, :, 6:9]	= images[i + 1]
			input_frames[0, :, :, 9:]	= images[i + 2]
			gen_frame = runner.g_model.generate_image(input_frames)[0]
			mae = mean_absolute_error(gen_frame, images[i])
			print(mae)
	
			split_name  = os.path.splitext(image_paths[i])
			name = split_name[0] + "_gen" + split_name[1]
			cv2.imwrite(name, gen_frame)

def mean_absolute_error(A, B):
	floatA = A.astype(np.float)
	floatB = B.astype(np.float)
	mae = np.sum(np.absolute(floatB - floatA))
	mae /= np.prod(np.shape(floatA))
	return mae

if __name__ == '__main__':
	main()
