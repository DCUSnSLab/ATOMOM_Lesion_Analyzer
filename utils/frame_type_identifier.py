import cv2
import subprocess
import time

class FrameTypeIdentifier:
	def __init__(self, video_path):
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)
	def __print_error_message(self, message):
		print("\033[91m" + message + "\033[0m")

	# 변경된 identify_frame_types 함수
	def identify_frame_types(self):
		if not self.cap.isOpened():
			self.__print_error_message("Error: Unable to open video file\n\tvideo_path:" + self.video_path)
			exit()

		frame_types = self.get_frame_types()

		frame_index = 0
		lis = []
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			frame_type = frame_types[frame_index]
			if(frame_type.split(',')[2] == 'I'):
				print(f"Frame {frame_index + 1} type: {[frame_type]}")
				lis.append(frame_index + 1)
			frame_index += 1
		self.cap.release()

	# 새로운 get_frame_types 함수
	def get_frame_types(self):
		# cmd = ['ffprobe', '-loglevel', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pict_type', '-of',
		#        'csv=p=0', self.video_path]
		cmd = ['ffprobe', '-select_streams', 'v', '-show_entries', 'frame=pkt_pts_time,pict_type', '-of',
		       'csv', self.video_path]
		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		frame_types = result.stdout.strip().split('\n')
		return frame_types


if __name__ == '__main__':
	video_path = '/home/dgdgksj/ATOMOM_Lesion_Analyzer/test_data/atomom_test_videos/test_unknown5.mp4'
	start_time = time.time()
	frame_identifier = FrameTypeIdentifier(video_path)
	frame_identifier.identify_frame_types()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Elapsed Time: {elapsed_time:.2f} seconds")
