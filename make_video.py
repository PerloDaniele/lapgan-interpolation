from avg_runner import AVGRunner
import constants as c
import getopt
import sys
import os
import cv2
import numpy as np
from copy import deepcopy
from scipy.misc import imsave

def main():
	try:
		opts, _ = getopt.getopt(sys.argv[1:], 'l:v', ['load_path=', 'video='])
	except getopt.GetoptError:
		print('Invalid parameters.')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-l', '--load_path'):
			load_path = arg
		if opt in ('-v', '--video'):
			input_video = arg
			assert os.path.exists(input_video)
	
	do_the_thing(load_path, input_video)

def do_the_thing(load_path, video):
	stream	 	= cv2.VideoCapture(video)
	width	 	= int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	height		= int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps 	 	= int(stream.get(cv2.CAP_PROP_FPS))
	fourcc	 	= int(stream.get(cv2.CAP_PROP_FOURCC))
	frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

	split_name  = os.path.splitext(video)
	output_name = split_name[0] + "_out" + split_name[1]
	video 		= cv2.VideoWriter(output_name, fourcc, fps * 2, (width, height))

	c.FULL_HEIGHT = height
	c.FULL_WIDTH  = width
	runner = AVGRunner(0, load_path)

	input_frames 	= np.empty([1, c.FULL_HEIGHT, c.FULL_WIDTH, 3 * c.HIST_LEN], dtype=np.uint8)
	history_frames	= np.empty([c.HIST_LEN, c.FULL_HEIGHT, c.FULL_WIDTH, 3], dtype=np.uint8)
	
	i = 0
	frame_num = 1
	while True:
		success, frame = stream.read()
		if not success:
			break
		print("%d/%d" % (frame_num, frame_count))

		history_frames[i] = frame
		i = (i + 1) % c.HIST_LEN

		for input_index in range(c.HIST_LEN):
			input_frames[0, :, :, 3 * (input_index):3 * (input_index + 1)] = history_frames[(i + input_index) % c.HIST_LEN]
		
		gen_frame = runner.g_model.generate_image(input_frames)[0]
		video.write(history_frames[(i + 1) % c.HIST_LEN])
		video.write(gen_frame)
		frame_num += 1
	video.release()
	stream.release()

if __name__ == '__main__':
	main()
