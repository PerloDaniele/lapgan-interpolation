from pytube import YouTube
from multiprocessing import cpu_count
from time import sleep
from glob import glob
import threading
import math
import sys
import os

def main():
	if len(sys.argv) < 1:
		sys.exit(1)

	video_list_file = sys.argv[1]
	urls = []
	with open(video_list_file) as file:
		for line in file:	
			url = line.split()[0]
			urls.append(url)

	thread_num = cpu_count() * 2
	threads = []
	urls_num = len(urls)
	urls_per_thread = int(math.ceil(urls_num / thread_num))
	for i in range(thread_num):
		log_good = open("logs/log_good_{}.txt".format(i), 'w', 1)
		log_bad = open("logs/log_bad_{}.txt".format(i), 'w', 1)
		urls_slice = urls[i * urls_per_thread : min(urls_num, (i + 1) * urls_per_thread)]
		new_thread = threading.Thread(target=check_videos, args=(urls_slice, log_good, log_bad))
		new_thread.start()
		threads.append(new_thread)

	for thread in threads:
		thread.join()

	logs_good = glob("logs/log_good_*.txt")
	with open("logs/log_good.txt", 'w') as log_file:
		for log in logs_good:
			with open(log) as file:
				for line in file:
					log_file.write(line)
			os.remove(log)
	logs_bad = glob("logs/log_bad_*.txt")
	with open("logs/log_bad.txt", 'w') as log_file:
		for log in logs_bad:
			with open(log) as file:
				for line in file:
					log_file.write(line)
			os.remove(log)

def check_videos(urls, log_good, log_bad):
	for url in urls:
		try:
			yt = YouTube(url)
			log_good.write(url + "\n")
		except:
			log_bad.write(url + "\n")
		print(url)
	log_good.close()
	log_bad.close()

if __name__ == '__main__':
	main()