import numpy as np
import cv2

	

class Memory:
	def __init__(self, memory_size = 100):
		self.memory = []
		self.memory_size = memory_size	

	def remember(self, state, a, next_state, r):
		self.memory.append({'state': state, 'action':a, 'next_state': next_state, 'r': r})
		if len(self.memory) > self.memory_size:
			self.memory.pop(0)
	
	def batch(self, model, batch_size, gama = 0.9):
		print 'batch'

class VideoContainer:
	def __init__(self, video_dir = 'data_debug', video_list = 'video_list', 
			height = 256, width = 512, n_channels = 3, max_window_size = 1500, 
			step = 10, context_ratio = 0.2, alpha = 0.1, eta = 0.3, tau = 0.6):
		video_list = open(video_dir + '/' + video_list, 'r').read().split()
		print video_list
		self.videos = []
		for v in video_list:
			v = VideoProcessor(v)
			self.videos.append(v)

		self.width = width
		self.height = height
		self.n_channels = n_channels
		self.max_window_size = max_window_size
		self.context_ratio = context_ratio
		self.alpha = alpha
			
	
	def extractFrames(self, i, start, end, step = 10):
		offset = int(np.floor((end - start) * self.context_ratio))
		start -= offset
		end += offset
	
		if start < 0: start = 0
		if end > self.videos[i].length: end = length
		if end - start > self.max_window_size: end = start + self.videos[i].max_window_size 
		
		ret = self.videos[i].frames(start, end, step, self.height, self.width, self.n_channels)
		print ret
		return ret		

	def play_one_step(self, state, action, gt):
		id = state[0]
		start = state[1]
		end = state[2]

		gt_start = gt[0]
		gt_end = gt[1]
		length = end - start
		
		print action	
	
		# Transfer to a new state
		if action == 'move_left':
			alpha = - self.alpha
			new_start = start + alpha * length
			new_end = end + alpha * length
		
		elif action == 'move_right':
			alpha = self.alpha
			new_start = start + alpha * length
			new_end = end + alpha * length

		elif action == 'resize_enlarge':
			alpha = self.alpha
			new_start = start - alpha * length
			new_end = end + alpha * length
		
		elif action == 'resize_shrink':
			alpha = self.alpha
			new_start = start + alpha * length
			new_end = end - alpha * length
		else:
			new_start = start
			new_end = end
					
		# Obtain reward
		if action != 'trigger':
			I = max(0, min(end, gt_end) - max(start, gt_start))
			U = max(end, gt_end) - min(start, gt_start)
			IoU = I * 1.0 / U

			I = max(0, min(new_end, gt_end) - max(new_start, gt_start))
			U = max(new_end, gt_end) - min(new_start, gt_start)
			new_IoU = I * 1.0 / U
			r = np.sign(new_IoU - IoU)
		else:
			I = max(0, min(end, gt_end) - max(start, gt_start))
			U = max(end, gt_end) - min(start, gt_start)
			IoU = I / U

			if IoU > tau:
				r = eta
			else:
				r = -eta

		
		if new_start < 0: new_start = 0
		if new_end > self.videos[id].length: new_end = self.videos[id].length
	
		state.append(id)
		state.append(new_start)
		state.append(new_end)

		return state, r
		
	
class VideoProcessor:
	def __init__(self, video_path = 'data_debug/10_0_c.avi'):
		self.cap = cv2.VideoCapture(video_path)
		self.length = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

		
	def frames(self,  start, end, step = 10, height = 256, width = 512, n_channels = 3, max_window_size = 1500):
		cap = self.cap
	
#		if start < 0: start = 0
#		if end > video_length: end = video_length
#		if end - start > self.max_window_size: end = start + self.max_window_size 
		ret = np.zeros((max_window_size, n_channels, height, width))		

		print start, end, step
		count = 0
		for i in xrange(start, end , step):
			cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
			retval, frame = cap.read()
			frame = cv2.resize(frame, (width, height))
			for c in xrange(0, n_channels):
				ret[count, c, :, :] = frame[:, :, c]
			count += 1
		return ret
		
		
if __name__ == '__main__':
	c = VideoContainer()
	c.extractFrames(0, 10, 50)
	state, r = c.play_one_step([0, 10, 20], 'move_right', [15, 25])
	print state, r
	
	m = Memory()
	m.remember([0, 10, 20], 'trigger', [0, 15, 25], 1)
