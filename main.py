from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense
from keras. layers.wrappers import TimeDistributed
import VladLayer
#constructure network
input_sequenses = Input(shape = (15, 3, 512, 256))
base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

#model =  Dense(output_dim = 128, input_dim = 512)
output =  TimeDistributed(model)(input_sequenses)

