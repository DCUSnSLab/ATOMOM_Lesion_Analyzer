import cv2
import time
# try:
# 	from cv2 import cv2
# except ImportError:
# 	pass


class VideoFrameSampler:
	def __init__(self):
		self.selected_frames = []
		self.frame_indices = []
	def __print_error_message(self, message):
		print("\033[91m" + message + "\033[0m")  # ANSI escape code for red text

	def sample_frames(self, video_path, num_samples):
		# note
		#   total_frame = 467
		#   num_samples = 10
		#   interval = total_frames // num_samples = 467 // 10 = 46
		#   frame_indices = [0, 46, 92, 138, 184, 230, 276, 322, 368, 414]
		cap = cv2.VideoCapture(video_path)

		if not cap.isOpened():
			self.__print_error_message("Error: Unable to open video file\n\tvideo_path:" + video_path, )
			exit()
			return self.selected_frames

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		if num_samples == -1:
			frame_indices = range(total_frames)
		else:
			frame_indices = [i * (total_frames // num_samples) for i in range(num_samples)]

		for frame_index in frame_indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
			ret, frame = cap.read()
			if ret:
				self.selected_frames.append(frame)
			else:
				break
		cap.release()

		return self.selected_frames

	def display_frames(self,wait_time=10):
		win_name = 'Selected Frame'
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

		for frame in self.selected_frames:
			cv2.imshow(win_name, frame)
			cv2.resizeWindow(winname=win_name, width=640, height=480)
			cv2.waitKey(wait_time)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	start_time = time.time()
	video_path = '/home/dgdgksj/ATOMOM_Lesion_Analyzer/test_data/atomom_test_videos/test_miso1.mp4'
	num_samples = 100
	sampler = VideoFrameSampler()
	sampler.sample_frames(video_path, num_samples)
	sampler.display_frames()
	print("elapsed time", time.time() - start_time)
# sampler.release_video()
