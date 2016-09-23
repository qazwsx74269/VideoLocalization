import numpy as np
import VideoContainer.py
from keras.applications.resnet50 import ResNet50

class Memory:
	def __init__(self, memory_size = 100):
		self.memory = []
		self.memory_size = memory_size	

	def remember(self, state, a, next_state, r):
		self.memory.append({'state': state, 'action':a, 'next_state': next_state, 'r': r})
		if len(self.memory) > self.memory_size:
			self.memory.pop(0)
	
	def batch(self, model, batch_size, gama = 0.9):
		print 'memory batch'

class Model:
	def __init__(self, height = 256, width = 512, n_channels = 3, max_window_size = 1500, step = 10):
		print 'model'
		self.height = height
		self.width = width
		self.n_channels = n_channels
		self.max_window_size = max_window_size
		self.step = step
		
	
	def constructNetwork():
		res_net = ResNet50(weights='imagenet')

if __name__ == '__main__':
	model = Model()
	
		
