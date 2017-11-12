import getopt
import sys
import os
import cv2
import numpy as np
from scipy.misc import imsave

def main():
	try:
		opts, _ = getopt.getopt(sys.argv[1:], 'v:l', ['video=', 'length='])
	except getopt.GetoptError:
		print('Invalid parameters.')
		sys.exit(2)

	length = 10 #sec
	for opt, arg in opts:
		if opt in ('-v', '--video'):
			video = arg
		if opt in ('-l', '--length'):
			lenght = arg

	if not os.path.exists(video):
		sys.exit(2)
	if length <= 0:
		sys.exit(2)
	
	stream	 	= cv2.VideoCapture(video)
	width	 	= int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	height		= int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps 	 	= int(stream.get(cv2.CAP_PROP_FPS))
	frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
	fourcc	 	= int(cv2.VideoWriter_fourcc(*'MJPG'))

	frames_to_process = fps * length
	first_frame = np.random.randint(frame_count - frames_to_process)
	stream.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

	split_name  = os.path.splitext(video)
	output_name = split_name[0] + "_" + str(first_frame) + "_" + str(length) + split_name[1]
	video 		= cv2.VideoWriter(output_name, fourcc, fps, (width, height))

	for i in range(frames_to_process):
		print("%d/%d" % (i+1, frames_to_process))
		success, frame = stream.read()
		if not success:
			break
		video.write(frame)
	video.release()

if __name__ == '__main__':
	main()
