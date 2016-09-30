from keras import backend as K
from keras.engine import Layer, InputSpec
import numpy as np
from keras import initializations, activations
from keras.layers import Input, Dense, core
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
import theano

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
		self.c = self.c_init((self.n_centers, self.d), name = '{}_c'.format(self.name))

		self.zero = K.zeros((1,))
	
		self.trainable_weights = [self.w, self.b, self.c]

		self.softmax = activations.get('softmax')
	
	def compute_mask(self, input, input_mask = None):
		return None


	def call(self, x, mask = None):
		if mask is not None:
			print 'foo'
			mask = K.any(mask, axis = -1)
			mask = K.repeat_elements(K.expand_dims(mask, 2), self.n, axis = 2)
			mask = K.reshape(mask, (-1, self.l * self.n))
			mask = K.expand_dims(mask, 2)

		x = K.reshape(x, (-1, self.l * self.n, self.d))	
		a = self.softmax(K.dot(x, self.w) + self.b)
		if mask is not None:
			a = K.switch(mask, a, 0)
		a = a.dimshuffle((2, 0, 1))
		a = K.expand_dims(a, 3)
		
		bias = K.expand_dims(self.c, 1)
		bias = K.expand_dims(bias, 1)
		x_bar = K.expand_dims(x, 0)

		def _step(a, c, x):
			output = K.sum(a * (x - c), axis = 1)
			return output
			
		ret, _ = theano.scan(fn = _step, 
					outputs_info = None, 
					sequences = [a, bias], 
					non_sequences = x)

		ret = ret.dimshuffle((1, 0, 2))
			
		return ret
#		return mask

		

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.d, self.n_centers)
#		return (self.n_centers, input_shape[0], self.l * self.d)
#		return (input_shape[0], self.l * self.n, self.d, self.n_centers)
#		return (input_shape[0], self.l * self.n)

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
	
		





