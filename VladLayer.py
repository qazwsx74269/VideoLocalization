from keras import backend as K
from keras.engine import Layer, InputSpec
import numpy as np
from keras import initializations, activations
from keras.layers import Input, Dense, core
from keras.models import Model
from keras.layers.wrappers import TimeDistributed


class VladLayer(Layer):
	def __init__(self, n_centers, w_init = 'glorot_uniform', b_init = 'zero', c_init = 'zero',  **kwargs):
		self.n_centers = n_centers
		self.w_init = initializations.get(w_init)
		self.b_init = initializations.get(b_init)
		self.c_init = initializations.get(c_init)
		
		super(VladLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(dtype = K.floatx(), shape = input_shape)]
		self.n = input_shape[1]
		self.d = input_shape[2]

		self.w = self.w_init((self.d, self.n_centers), name = '{}_w'.format(self.name))
		self.b = self.b_init((self.n_centers, ), name = '{}_b'.format(self.name))
		self.c = self.c_init((self.d, self.n_centers), name = '{}_c'.format(self.name))
		
		initial_c_m = np.repeat(np.eye(input_shape[2]), self.n_centers, axis = 0)
		self.c_m = K.variable(initial_c_m)

		self.trainable_weights = [self.w, self.b, self.c]

		self.softmax = activations.get('softmax')

	def call(self, x, mask=None):
		input_shape = self.input_spec[0].shape

		a = self.softmax(K.dot(x, self.w))

		a = K.repeat_elements(K.expand_dims(a, 2), self.d, axis = 2)
		
		stack_x = K.repeat_elements(K.expand_dims(x, 3), self.n_centers, axis = 3)
		bias = K.repeat_elements(K.expand_dims(self.c, 0), self.n , axis = 0)
		diff = stack_x - bias
		
		ret = a * diff
		ret = K.sum(ret, axis = 1)
		
		return ret
#		return diff

		

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.n_centers, self.d)
#		return (input_shape[0], self.n, self.d, self.n_centers)


if __name__ == '__main__':
	data = np.random.random((3, 15, 25))
	
	
	input_sequences = Input(shape = (15, 25))
	h = TimeDistributed(Dense(output_dim = 30))(input_sequences)
	h = VladLayer(n_centers = 5)(h)
#	h = core.Reshape((25 * 5, ))(h)
	model = Model(input = input_sequences, output = h)
	model.compile(optimizer='rmsprop',
	              loss='categorical_crossentropy',
        	      metrics=['accuracy'])
	print 'compile complete...'
	res = model.predict(data)
	print res.shape
	
		





