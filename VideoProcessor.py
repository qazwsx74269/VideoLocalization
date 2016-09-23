import cv2
import numpy as np

class VideoProcessor:
	def __init__(self, video_path = 'data_debug/10_0_c.avi',  height = 256, width = 512, n_channels = 3, max_window_size = 1500):
		self.cap = cv2.VideoCapture(video_path)
		self.video_length = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
		self.height = height
		self.width = width
		self.n_channels = n_channels
		self.max_window_size = max_window_size

		
	def frames(self,  start, end, step = 10):
		cap = self.cap
		video_length = self.video_length
	
#		if start < 0: start = 0
#		if end > video_length: end = video_length
#		if end - start > self.max_window_size: end = start + self.max_window_size 
		ret = np.zeros((self.max_window_size, self.n_channels, self.height, self.width))		

		print start, end, step
		count = 0
		for i in xrange(start, end , step):
			cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
			retval, frame = cap.read()
			frame = cv2.resize(frame, (self.width, self.height))
			for c in xrange(0, self.n_channels):
				ret[count, c, :, :] = frame[:, :, c]
			count += 1
		return ret

	
	def play(self, state, action, gt):
		start = state[0]
		end = state[1]

		gt_start = gt[0]
		gt_end = gt[1]
		length = end - start
		
		# Obtain reward

		# Transfer to a new state
		if action['name'] == 'move_left':
			alpha = - self.alpha
			start += alpha * length
			end += alpha * length

		elif action['name'] == 'move_right':
			alpha = self.alpha
			start += alpha * length
			end += alpha * length

		elif action['name'] == 'resize_enlarge':
			alpha = self.alpha
			start -= alpha * length
			end += alpha * length
		
		elif action['name'] == 'resize_shrink':
			alpha = self.alpha
			start += alpha * length
			end -= alpha * length
		
		
		if end > video_length: end = video_length
		
	
	
	
if __name__ == '__main__':
	p = VideoProcessor()
	p.frames(10, 50)
