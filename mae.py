from avg_runner import AVGRunner
import constants as c
import getopt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import cv2
from glob import glob
from sklearn.metrics import mean_absolute_error
from utils import normalize_frames

def main():
	
	input_mean = False
	
	try:
		opts, _ = getopt.getopt(sys.argv[1:], 'l:d:m:na', ['load_path=', 'directory=', 'mean=', 'no_adversarial'])
	except getopt.GetoptError:
		print('Invalid parameters.')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-l', '--load_path'):
			load_path = arg
		if opt in ('-d', '--directory'):
			directory = arg
		if opt in ('-m', '--mean'): #TO DO: BUG DO NOT USE IT
			input_mean = True
			mean_file = arg
		if opt in ('-na', '--no_adversarial'):
			c.ADVERSARIAL = False

	if not os.path.exists(directory):
		print('Dir not found.')

	
	if input_mean:
		mean = np.load(mean_file)['arr_0']
        
	image_paths = sorted(glob(os.path.join(directory, '*')))
	shape = np.shape(cv2.imread(image_paths[0]))
	c.FULL_HEIGHT = shape[0]
	c.FULL_WIDTH  = shape[1]
	images = np.empty([len(image_paths), c.FULL_HEIGHT, c.FULL_WIDTH, 3])
	for i, image in enumerate(image_paths):
		out_image = cv2.imread(image)
		
		#out_image = out_image[:,:,[2,1,0]]
		
		images[i, :, :, :] = out_image#[:,:,[2,1,0]]
		#images[i, :, :, :] = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
	
	if input_mean:
		images = np.clip((images - mean),-1,1)

	runner = AVGRunner(0, load_path)
	model_name = os.path.basename(load_path)
	dir_out = get_dir('../interpolated_frames/' + model_name)
	
	input_frames = np.empty([1, c.FULL_HEIGHT, c.FULL_WIDTH, 3 * c.HIST_LEN], dtype=np.uint8)
	h = c.HIST_LEN / 2
	n = c.HIST_LEN
	for i in range(h, len(images) - h):
		for j in range(h):
			input_frames[0, :, :, j*3:(j+1)*3] = images[i - h + j]
			input_frames[0, :, :, (n - j - 1)*3:(n - j)*3] = images[i + h - j]

		
		gen_frame = runner.g_model.generate_image(input_frames)[0]
		mae = mean_absolute_error(gen_frame, images[i])
		print(mae)

		split_name  = os.path.splitext(os.path.basename(image_paths[i]))
		name = os.path.join(dir_out, split_name[0] + "_gen" + split_name[1])

		if input_mean:
			gen_frame = np.clip((gen_frame + mean),-1,1)

		cv2.imwrite(name, gen_frame)
		
		'''
		batch = np.empty([1, c.FULL_HEIGHT, c.FULL_WIDTH, 3 * (c.HIST_LEN + 1)])
		batch[0, :, :, :3 * c.HIST_LEN] = normalize_frames(input_frames)
		batch[0, :, :, 3 * c.HIST_LEN:] = normalize_frames(images[i])
		runner.g_model.test_batch(batch, 420)
		'''

def mean_absolute_error(A, B):
	floatA = A.astype(np.float)
	floatB = B.astype(np.float)
	mae = np.sum(np.absolute(floatB - floatA))
	mae /= np.prod(np.shape(floatA))
	return mae

def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

if __name__ == '__main__':
	main()
