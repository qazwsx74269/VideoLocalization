from keras import backend as K
from keras.engine import Layer, InputSpec
import numpy as np
from keras import initializations, activations
from keras.layers import Input, Dense, core
from keras.models import Model
from keras.layers.wrappers import TimeDistributed


class VladLayer(Layer):
	def __init__(self, n_centers, w_init = 'glorot_uniform', b_init = 'zero', c_init = 'zero',  **kwargs):
		self.supports_masking = True
		self.n_centers = n_centers
		self.w_init = initializations.get(w_init)
		self.b_init = initializations.get(b_init)
		self.c_init = initializations.get(c_init)
		
		super(VladLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(dtype = K.floatx(), shape = input_shape)]
		self.l = input_shape[1]
		self.n = input_shape[2]
		self.d = input_shape[3]

		self.w = self.w_init((self.d, self.n_centers), name = '{}_w'.format(self.name))
		self.b = self.b_init((self.n_centers, ), name = '{}_b'.format(self.name))
		self.c = self.c_init((self.d, self.n_centers), name = '{}_c'.format(self.name))
		
		self.trainable_weights = [self.w, self.b, self.c]

		self.softmax = activations.get('softmax')

	def call(self, x, mask = None):
		print mask
		if mask is not None:
			print 'foo'
			#mask = K.repeat_elements(K.expand_dims(mask, 2), self.n, axis = 2)
			#mask = K.reshape(mask, (-1, self.l * self.n))		

		x = K.reshape(x, (-1, self.l * self.n, self.d))	

		a = self.softmax(K.dot(x, self.w))
		a = K.repeat_elements(K.expand_dims(a, 2), self.d, axis = 2)
		
		stack_x = K.repeat_elements(K.expand_dims(x, 3), self.n_centers, axis = 3)
		bias = K.repeat_elements(K.expand_dims(self.c, 0), self.n * self.l , axis = 0)
		diff = stack_x - bias
			
		ret = a * diff
		ret = K.sum(ret, axis = 1)
		
				
			
#		return ret
		return mask

		

	def get_output_shape_for(self, input_shape):
#		return (input_shape[0], self.n_centers, self.d)
#		return (input_shape[0], self.n, self.d, self.n_centers)
#		return (input_shape[0], self.l * self.n, self.d, self.n_centers)
		return (input_shape[0], self.l * self.n)

if __name__ == '__main__':
	data = np.random.random((3, 5, 15, 25))
		
	input_sequences = Input(shape = (5, 15, 25))
	#dense_time = TimeDistributed(Dense(output_dim = 30))
	#h = dense_time(input_sequences)
	#print dense_time.output_shape
	h = VladLayer(n_centers = 5)(input_sequences)
#	h = core.Reshape((25 * 5, ))(h)
	model = Model(input = input_sequences, output = h)
	res = model.predict(data)
	print res.shape
	
		





