from pytube import YouTube
from time import sleep
from glob import glob
import math
import sys
import os

def main():
	if len(sys.argv) < 1:
		sys.exit(1)

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

if __name__ == '__main__':
	main()