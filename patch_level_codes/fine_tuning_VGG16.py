import numpy as np
import os
import shutil
import sys
import random
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2DTranspose, Conv2D, BatchNormalization
from keras import applications,optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping
import cv2
import tensorflow as tf
import datetime
import keras
import pickle
from multiprocessing import Lock
from multiprocessing.managers import BaseManager


# K.set_image_dim_ordering('th')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# # parameters
# cross_validation_id = 0
# fixed_address = '{0}/..'.format(os.path.dirname(os.path.realpath(__file__)))
# # fixed_address = '../../../ramdiskmp'
# setting = '2598_{0}'.format(cross_validation_id)
# # setting = '2584'

# eps  = 50
# final_model_epochs = 30
# img_width, img_height = 256, 256

# top_model_path = '{0}/bottleneck_fc_model_{1}.h5'.format(fixed_address,setting)

# final_model_path = '{0}/final_model_{1}.h5'.format(fixed_address,setting)
# # final_model_path = '../results/2580/Final_Models/final_model_{0}.h5'.format(setting)

# bottleneck_features_training_path = '/space/ariyanzarei/bottleneck_features_training_2595_{0}.npy'.format(cross_validation_id)
# bottleneck_features_validation_path = '/space/ariyanzarei/bottleneck_features_validation_2595_{0}.npy'.format(cross_validation_id)

# # bottleneck_features_training_path = '/extra/ariyanzarei/bottleneck_features_training_{0}.npy'.format(setting)
# # bottleneck_features_validation_path = '/extra/ariyanzarei/bottleneck_features_validation_{0}.npy'.format(setting)

# data_path = '/space/ariyanzarei/complete_dataset_no_duplicate_{0}'.format(cross_validation_id)
# # data_path = '{0}/data/20p_hyper_dataset_divided'.format(fixed_address)
# train_data_dir = '{0}/training'.format(data_path)
# validation_data_dir = '{0}/validation'.format(data_path)
# test_data_dir = '{0}/test'.format(data_path)

# numbers_tr = [225792,240192,226368,241152,241344]
# numbers_va = [72960,72960,72960,72960,72192]
# numbers_te = [82368,68160,81984,67200,67584]

# nb_train = numbers_tr[cross_validation_id]
# nb_train_normal = int(nb_train/2)
# nb_train_abnormal = int(nb_train/2)
# nb_validation = numbers_va[cross_validation_id]
# nb_validation_normal = int(nb_validation/2)
# nb_validation_abnormal = int(nb_validation/2)
# nb_test = numbers_te[cross_validation_id]
# nb_test_normal = int(nb_test/2)
# nb_test_abnormal = int(nb_test/2)

# batch_size = 32

#  -------------------------------

# K.set_image_dim_ordering('th')

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# fixed_address = '{0}/..'.format(os.path.dirname(os.path.realpath(__file__)))
# setting = '2599'

# eps  = 50
# final_model_epochs = 30
# img_width, img_height = 256, 256

# top_model_path = '{0}/bottleneck_fc_model_{1}.h5'.format(fixed_address,setting)

# final_model_path = '{0}/final_model_{1}.h5'.format(fixed_address,setting)

# bottleneck_features_training_path = '/space/ariyanzarei/bottleneck_features_training_2598.npy'
# bottleneck_features_validation_path = '/space/ariyanzarei/bottleneck_features_validation_2598.npy'

# data_path = '/space/ariyanzarei/complete_dataset_full_A'
# train_data_dir = '{0}/training'.format(data_path)
# validation_data_dir = '{0}/test'.format(data_path)

# nb_train = 143296*2 # 148224 : 7
# nb_train_normal = int(nb_train/2)
# nb_train_abnormal = int(nb_train/2)
# nb_validation = 47744*2 # 42848 : 3
# nb_validation_normal = int(nb_validation/2)
# nb_validation_abnormal = int(nb_validation/2)

# batch_size = 32

def generalize_binary_crossentropy(yTrue,yPred,from_logits=False):

	a = -2
	b = -4.2	
	c = -1.9965

	main_loss = K.binary_crossentropy(yTrue,yPred,from_logits)
	
	gen_loss = (1-yTrue) * (a*(K.exp(-(b*(yPred+0.01))**2))-c) + (yTrue) * (a*(K.exp(-(b*(yPred-1.01))**2))-c)
	
	return  main_loss+gen_loss 

def save_bottlebeck_features():

	datagen = ImageDataGenerator(rescale=1. / 255)
	
	# build the VGG16 network
	model = applications.VGG16(include_top=False, weights='imagenet')

	generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	
	bottleneck_features_train = model.predict_generator(
		generator,nb_train//batch_size)
	np.save(open(bottleneck_features_training_path, 'wb'),
			bottleneck_features_train)

	print('>>> Setting:{0} > Bottleneck Features Save > features saved for the training set.'.format(setting))

	generator = datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	bottleneck_features_validation = model.predict_generator(
		generator,nb_validation//batch_size)
	np.save(open(bottleneck_features_validation_path, 'wb'),
			bottleneck_features_validation)
	
	print('>>> Setting:{0} > Bottleneck Features Save > features saved for the validation set.'.format(setting))


def train_top_model():
	train_data = np.load(open(bottleneck_features_training_path,'rb'))
	train_labels = np.array(
		[0] * (nb_train_abnormal) + [1] * (nb_train_normal))

	validation_data = np.load(open(bottleneck_features_validation_path,'rb'))
	validation_labels = np.array(
		[0] * (nb_validation_abnormal) + [1] * (nb_validation_normal))

	print('>>> Setting:{0} > Train Top Model > features loaded for both the train and validation sets.'.format(setting))

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='adam',loss=generalize_binary_crossentropy, metrics=['accuracy'])

	clbk = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=2, mode='auto', baseline=0.5, restore_best_weights=True)

	print('>>> Setting:{0} > Train Top Model > begining fitting the model.'.format(setting))

	history = model.fit(train_data, train_labels,
			  epochs=eps,
			  batch_size=batch_size,
			  callbacks=[clbk],
			  validation_data=(validation_data, validation_labels))

	print('>>> Setting:{0} > Train Top Model > model fitting finished.'.format(setting))

	model.save_weights(top_model_path)

	print('>>> Setting:{0} > Train Top Model > model weights saved.'.format(setting))

	with open('{0}/history_top_log_{1}.txt'.format(fixed_address,setting), 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	print('>>> Setting:{0} > Fine Tune > history log saved.'.format(setting))


def fine_tune_VGG16():

	vgg_model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(3, 256, 256))

	top_model = Sequential()
	top_model.add(Flatten(input_shape=(512,8,8)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	# top_model.add(Dense(128, activation='relu'))
	# top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))

	top_model.load_weights(top_model_path)
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
	# print(len(model.layers))
	# for layer in model.layers[:9]:
	#	 layer.trainable = False


	model.compile(loss=generalize_binary_crossentropy,
				  optimizer=optimizers.Adam(lr=1e-4),
				  metrics=['accuracy'])

	model.summary()

	train_datagen = ImageDataGenerator(
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode='binary')

	# clbk = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=2, mode='auto', baseline=0.5, restore_best_weights=True)
	clbk = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=2, mode='auto', baseline=0.5, restore_best_weights=True)

	print('>>> Setting:{0} > Fine Tune > model created and compiled.'.format(setting))

	print('>>> Setting:{0} > Fine Tune > begining fine-tuning process.'.format(setting))

	# change sample count 
	history = model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train//batch_size,
		epochs=final_model_epochs,
		callbacks=[clbk],
		validation_data=validation_generator,
	validation_steps=nb_validation//batch_size)

	print('>>> Setting:{0} > Fine Tune > fine-tuning process finished.'.format(setting))

	model.save_weights(final_model_path)

	print('>>> Setting:{0} > Fine Tune > final model weight saved.'.format(setting))

	with open('{0}/history_full_log_{1}.txt'.format(fixed_address,setting), 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	print('>>> Setting:{0} > Fine Tune > history log saved.'.format(setting))


def load_full_model():
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

def load_full_model_outside_use(final_model_path_param):
	vgg_model = applications.VGG16(include_top=False,input_shape=(3, 256, 256),)
	print('Model loaded.')

	top_model = Sequential()
	top_model.add(Flatten(input_shape=(512,8,8)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))

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

	model.load_weights(final_model_path_param)
	model.compile(loss='binary_crossentropy',
				  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
				  metrics=['accuracy'])
	return model

def predict_img(img,mdl):

	test_datagen = ImageDataGenerator(rescale=1. / 255)
	img = img_to_array(img)
	img = np.expand_dims(img,axis=0)
	test_generator = test_datagen.flow(img)
	lbl = mdl.predict_generator(test_generator,steps=1)

	# img = img_to_array(img)/255.
	# img = img[0:3,0:256,0:256]
	# img = np.expand_dims(img,axis=0)
	# lbl = mdl.predict(img)

	return lbl
	# return mdl.predict(np.expand_dims(np.reshape(np.array(img),(3,img_width,img_height)),axis=0)/255.)

def predict_batch(location,mdl,numb):

	files =  os.listdir(location)
	randomely_chosen = random.sample(files,numb)

	for f in randomely_chosen:
		img = cv2.imread(location+'/'+f)
		print('{0} -> {1}'.format(f,predict_img(img,mdl)))


# # redirect std out to a file
# original = sys.stdout
# sys.stdout = open('{0}/log_{1}.txt'.format(fixed_address,setting), 'w')

# # save_bottlebeck_features()
# # train_top_model()
# fine_tune_VGG16()

# # redirect std out back to its original form
# sys.stdout = original
