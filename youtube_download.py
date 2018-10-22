#!/usr/bin/env python
# import library
from pytube import YouTube
import os

# Initialize a global counter
counter = 0

# Show the download progress bar
def progress_check(stream, chunk, file_handle, remaining):
	'''
	This function takes in default 4 parameters generated automatically by pytube
	and show the download progress in percentage
	'''
	global counter

	# Gets the percentage of the file that has been downloaded.

	# Calculate the percentage of downloaded
	percent = (100*(file_size-remaining))/file_size
	# Only show the progress for every 1%
	if int(percent) != counter:
		print("{0:.1f}% downloaded".format(percent))
		# Increment the counter for downloaded percentage
		counter += 1

# Default download file location
def file_path():
	'''
	This function finds the complete path for Downloads folder

	/Users/@username/Downloads
	'''

	# If a download path has already been chosen by the user
	if path:
		return path
	# If no download path is specified
	else:
		# Find the prefix of the download path: /Users/@username
	    home = os.path.expanduser('~')
	    download_path = os.path.join(home, 'Downloads')
	    return download_path

def download():
	'''
	The main function for downloading YouTube video
	'''

	# User-input for download file path
	global path
	path = input('Download file location:')

	print("Download video will be saved to: {}".format(file_path()))

	# User-input for youtube video link
	link = input('Paste your YouTube video URL: ')
	print(link)

	print('Wielding the staff....')

	try:
		# YouTube object generated from the link
		yt = YouTube(link, on_progress_callback=progress_check)
	except:
		# Failed to create the YouTube object
		print('Magic does not always work.... :(\nCheck if you put a correct YouTube URL')
		redo = download()

	# Check files that: 
	# 1. Have both audio and video together in one file (progressive files)
	# 2. File is in mp4 format
	# 3. Select the one with the highest possible resolution
	video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

	print('Fetching: {}...'.format(yt.title))

	# Declare a global variable: file_size
	global file_size
	# Get the total file size of this video
	file_size = video.filesize
	# Download the file to target file path
	video.download(file_path())

	print('Download is complete!')

# Run the main function
download()





