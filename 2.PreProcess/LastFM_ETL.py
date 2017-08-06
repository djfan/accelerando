import os, csv, json, glob, pandas as pd

class MillionSong:
	def file_traverse(self, directory, csv_directory):
		for dirName, subdirList, fileList in os.walk(directory, topdown=False):
			for fname in fileList:
				file_path = dirName + '/' + fname
				with open(file_path) as json_file:
					if "json" in file_path:
						data = json.load(json_file) # data is dic type
						# print(data)
						with open("result.csv", 'a', newline='') as csvfile:
							tmp = [data["artist"], data["track_id"], data["title"]]
							spamwriter = csv.writer(csvfile, delimiter=',')
							spamwriter.writerow(tmp)


try:
	file = open("result.csv", 'r')
except IOError:
	file = open("result.csv", 'w')

current_directory = os.getcwd()
csv_directory = current_directory + "/result.csv"
example_directory = current_directory + "/lastfm_subset"
million_song = MillionSong()
million_song.file_traverse(example_directory, csv_directory)
# print(cwd + "/lastfm_subset")


