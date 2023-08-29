import cv2
import time


class VideoFrameSampler:
	@staticmethod
	def display_frame(frame, wait_time=10):
		win_name = 'Selected Frame'
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.imshow(win_name, frame)
		cv2.resizeWindow(winname=win_name, width=640, height=480)
		cv2.waitKey(wait_time)

	@staticmethod
	def sample_frames(video_path, num_samples):
		cap = cv2.VideoCapture(video_path)

		if not cap.isOpened():
			print("Error: Unable to open video file\n\tvideo_path:", video_path)
			exit()

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		if num_samples == -1:
			frame_indices = range(total_frames)
		else:
			frame_indices = [i * (total_frames // num_samples) for i in range(num_samples)]

		for frame_index in frame_indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
			ret, frame = cap.read()
			if ret:
				yield frame
			else:
				break

		cap.release()


if __name__ == "__main__":
	start_time = time.time()
	video_path = '/home/dgdgksj/ATOMOM_Lesion_Analyzer/test_data/atomom_test_videos/test_miso1.mp4'
	num_samples = 100
	sampler = VideoFrameSampler()
	for frame in sampler.sample_frames(video_path, num_samples):
		sampler.display_frame(frame)
		pass
	print("elapsed time", time.time() - start_time)
# sampler.release_video()
