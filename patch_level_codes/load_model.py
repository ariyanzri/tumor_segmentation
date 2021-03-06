from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2DTranspose, Conv2D, BatchNormalization
from keras import applications,optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping

K.set_image_dim_ordering('th')

def load_full_model(final_model_path):
	vgg_model = applications.VGG16(include_top=False,input_shape=(3, 256, 256))
	print('Model loaded.')

	top_model = Sequential()
	top_model.add(Flatten(input_shape=(512,8,8)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))

	# print(vgg_model.summary())
	# print(top_model.summary())

	model = Sequential()
	for layer in vgg_model.layers: 
		model.add(layer)
		if "conv" in layer.name:
			model.add(BatchNormalization())
	for layer in top_model.layers:
		model.add(layer)
		if "dense_1" in layer.name:
			model.add(BatchNormalization())
	
	# model.add(top_model)

	model.load_weights(final_model_path)
	model.compile(loss='binary_crossentropy',
				  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
				  metrics=['accuracy'])
	return model

model = load_full_model('/space/ariyanzarei/results/2598/final_model/final_model_2598.h5')
model.save('/work/ariyanzarei/mdl.h5')