import numpy as np
import os
import sys
# import fine_tuning_VGG16
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
# from keras.models import Sequential, load_model
# from keras.layers import Dropout, Flatten, Dense
# from keras import applications,optimizers
# from keras import backend as K
import multiprocessing

setting = '2598'

# K.set_image_dim_ordering('th')

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# # parameters
fixed_address = '{0}/..'.format(os.path.dirname(os.path.realpath(__file__)))

# img_width, img_height = 256, 256
# data_path = '/space/ariyanzarei/complete_dataset_full'
# train_data_dir = '{0}/training'.format(data_path)
# validation_data_dir = '{0}/test'.format(data_path)

# nb_train = 148224*2 # 143296 : 7
# nb_train_normal = int(nb_train/2)
# nb_train_abnormal = int(nb_train/2)
# nb_validation = 42848*2 # 47744 : 3
# nb_validation_normal = int(nb_validation/2)
# nb_validation_abnormal = int(nb_validation/2)

# batch_size = 32

final_model_path = '{0}/final_model_{1}.h5'.format(fixed_address,setting)

def all_set_performance_cv(model):
	datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = datagen.flow_from_directory(
	train_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')
	
	validation_generator = datagen.flow_from_directory(
	validation_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')

	test_generator = datagen.flow_from_directory(
	test_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')

	score_train = model.evaluate_generator(train_generator,nb_train/batch_size)
	score_validation = model.evaluate_generator(validation_generator,nb_validation/batch_size)
	score_test = model.evaluate_generator(test_generator,nb_test/batch_size)

	print('{0},{1},{2},{3},{4},{5}\n'.format(score_train[0],score_train[1],score_validation[0],score_validation[1],score_test[0],score_test[1]))

def all_set_performance(model):
	datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = datagen.flow_from_directory(
	train_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')
	
	validation_generator = datagen.flow_from_directory(
	validation_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')

	test_generator = datagen.flow_from_directory(
	test_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')

	score_train = model.evaluate_generator(train_generator,nb_train/batch_size)
	print('Train Scores [loss, accuracy]: {0}'.format(score_train))

	score_validation = model.evaluate_generator(validation_generator,nb_validation/batch_size)
	print('Validation Scores [loss, accuracy]: {0}'.format(score_validation))

	score_test = model.evaluate_generator(test_generator,nb_test/batch_size)
	print('Test Scores [loss, accuracy]: {0}'.format(score_test))

def test_set_performance(model):
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	batch_size=batch_size,
	target_size=(img_height, img_width),
	class_mode='binary')

	score = model.evaluate_generator(test_generator,nb_test/batch_size)
	# print(model.metrics_names)
	# scores = model.predict_generator(test_generator, nb_test/batch_size)
	print('[By Keras evaluate generator] The prediction accuracy and loss: {0}'.format(score))

def test_set_performance_single(directory_both,directory_save,mdl,thresh):
	files = os.listdir(directory_both+'/normal')
	files = [f for f in files if f.endswith('.jpg')]
	y_test = nb_test_normal*[1]+nb_test_abnormal*[0]
	
	true_normal = 0
	true_abnormal = 0

	i = 0
	for f in files:
		img = cv2.imread(directory_both+'/normal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		score = predict_img(img2,mdl)[0][0]
		if score<thresh:
			tmp = 0
		else:
			tmp = 1
		
		if tmp == y_test[i]:
			true_normal = true_normal +1
		else:
			cv2.imwrite(directory_save+'/FalseAbnormal/'+f,img)

		i+=1


	files = os.listdir(directory_both+'/abnormal')
	files = [f for f in files if f.endswith('.jpg')]

	for f in files:
		img = cv2.imread(directory_both+'/abnormal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		tmp = predict_img(img2,mdl)[0][0]
		if tmp<thresh:
			tmp = 0
		else:
			tmp = 1
		
		if tmp == y_test[i]:
			true_abnormal = true_abnormal +1
		else:
			cv2.imwrite(directory_save+'/FalseNormal/'+f,img)

		i+=1

	print('------------ Results by hand ------------')
	print('True normal: {0}'.format(true_normal))
	print('False normal: {0}'.format(nb_test_normal-true_normal))
	print('True abnormal: {0}'.format(true_abnormal))
	print('False abnormal: {0}'.format(nb_test_abnormal-true_abnormal))
	print('Threshold: {0}'.format(thresh))
	print('Normal accuracy: {0}'.format(true_normal/float(nb_test_normal)))
	print('Abnormal accuracy: {0}'.format(true_abnormal/float(nb_test_abnormal)))
	print('Total accuracy: {0}'.format((true_normal+true_abnormal)/(nb_test)))
	
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

def predict_all(directory_both,mdl):
	files = os.listdir(directory_both+'/normal')
	files = [f for f in files if f.endswith('.jpg')]
	y_test = nb_test_normal*[1]+nb_test_abnormal*[0]
	y_pred_keras = nb_test*[-1]
	image_names = nb_test*['empty']

	i = 0
	for f in files:
		img = cv2.imread(directory_both+'/normal/'+f)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		tmp = predict_img(img,mdl)[0][0]
		# print('{0} -> Predicted: {1}   True: {2}'.format(f,tmp,1))
		y_pred_keras[i] = tmp
		image_names[i] = 'N_{0}'.format(f)
		i+=1


	files = os.listdir(directory_both+'/abnormal')
	files = [f for f in files if f.endswith('.jpg')]

	for f in files:
		img = cv2.imread(directory_both+'/abnormal/'+f)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		tmp = predict_img(img,mdl)[0][0]
		# print('{0} -> Predicted: {1}   True: {2}'.format(f,tmp,0))
		y_pred_keras[i] = tmp
		image_names[i] = 'A_{0}'.format(f)
		i+=1


	# print(y_test)
	# print(y_pred_keras)
	

	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
	auc_keras = auc(fpr_keras, tpr_keras)
	print('Aread Under ROC Curve: {0}'.format(auc_keras))
	np.save('{0}/FP_rate_{1}'.format(fixed_address,setting),fpr_keras)
	np.save('{0}/TP_rate_{1}'.format(fixed_address,setting),tpr_keras)
	np.save('{0}/Thresholds_{1}'.format(fixed_address,setting),thresholds_keras)
	np.save('{0}/Image_names_{1}'.format(fixed_address,setting),image_names)

	# plt.figure(1)
	# plt.plot([0, 1], [0, 1], 'k--')
	# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	# plt.savefig('{0}/ROC_Curve_{1}.png'.format(fixed_address,setting))
	
def get_predictions(model):
	y_true_train = nb_train_normal*[1]+nb_train_abnormal*[0]
	y_pred_train  = np.zeros(nb_train)

	files = os.listdir(train_data_dir+'/normal')
	files = [f for f in files if f.endswith('.jpg')]

	i=0
	for f in files:
		img = cv2.imread(train_data_dir+'/normal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		score = predict_img(img2,model)[0][0]
		y_pred_train[i] = score
		i+=1

	print('training normal finished...')

	files = os.listdir(train_data_dir+'/abnormal')
	files = [f for f in files if f.endswith('.jpg')]

	for f in files:
		img = cv2.imread(train_data_dir+'/abnormal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		score = predict_img(img2,model)[0][0]
		y_pred_train[i] = score
		i+=1

	print('training abnormal finished...')

	# test

	y_true_validation = nb_validation_normal*[1]+nb_validation_abnormal*[0]
	y_pred_validation  = np.zeros(nb_validation)

	files = os.listdir(validation_data_dir+'/normal')
	files = [f for f in files if f.endswith('.jpg')]

	i=0
	for f in files:
		img = cv2.imread(validation_data_dir+'/normal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		score = predict_img(img2,model)[0][0]
		y_pred_validation[i] = score
		i+=1

	print('test normal finished...')

	files = os.listdir(validation_data_dir+'/abnormal')
	files = [f for f in files if f.endswith('.jpg')]

	for f in files:
		img = cv2.imread(validation_data_dir+'/abnormal/'+f)
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		score = predict_img(img2,model)[0][0]
		y_pred_validation[i] = score
		i+=1

	print('test abnormal finished...')

	return y_true_train,y_pred_train,y_true_validation,y_pred_validation

def get_predictions_and_save(test_patches_dir,resutls_path,model):
	slides = os.listdir(test_patches_dir)

	for s in slides:
		
		files_normal = os.listdir(test_patches_dir+'/'+s+'/normal')
		files_normal = [f for f in files_normal if f.endswith('.jpg')]
		files_abnormal = os.listdir(test_patches_dir+'/'+s+'/abnormal')
		files_abnormal = [f for f in files_abnormal if f.endswith('.jpg')]

		test_true = len(files_normal)*[1]+len(files_abnormal)*[0]
		test_pred  = np.zeros(len(test_true))

		i=0
		for f in files_normal:
			img = cv2.imread(test_patches_dir+'/'+s+'/normal/'+f)
			img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			score = predict_img(img2,model)[0][0]
			test_pred[i] = score
			i+=1

		print('test normal finished for slide {0}.'.format(s))

		for f in files_abnormal:
			img = cv2.imread(test_patches_dir+'/'+s+'/abnormal/'+f)
			img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			score = predict_img(img2,model)[0][0]
			test_pred[i] = score
			i+=1

		print('test abnormal finished for slide {0}.'.format(s))

		np.save('{0}/slide_{1}_true.txt'.format(resutls_path,s),test_true)
		np.save('{0}/slide_{1}_pred.txt'.format(resutls_path,s),test_pred)



def load_predictions_and_stats_test_patches(test_patches_dir,resutls_path):
	slides = os.listdir(test_patches_dir)
	cumulated_true = []
	cumulated_pred = []
	
	result = '' 

	for s in slides:
		
		files_normal = os.listdir(test_patches_dir+'/'+s+'/normal')
		files_normal = [f for f in files_normal if f.endswith('.jpg')]
		files_abnormal = os.listdir(test_patches_dir+'/'+s+'/abnormal')
		files_abnormal = [f for f in files_abnormal if f.endswith('.jpg')]

		test_true = len(files_normal)*[1]+len(files_abnormal)*[0]
		test_pred  = np.zeros(len(test_true))

		test_true = np.load('{0}/slide_{1}_true.txt.npy'.format(resutls_path,s))
		test_pred = np.load('{0}/slide_{1}_pred.txt.npy'.format(resutls_path,s))

		cumulated_true.extend(test_true)
		cumulated_pred.extend(test_pred)
		
		res_str = stats(test_true,test_pred)
		info_slide = 'Number slides Total(Normal,Abnormal): {0}({1},{2})'.format(len(files_normal)+len(files_abnormal),len(files_normal),len(files_abnormal))
		result = result + '------------------- RESULT SLIDE {0} -------------------\n{1}\n{2}'.format(s,info_slide,res_str)


	print(result)
	print('------ Cumulative results for the whole test set ------')
	print(stats(np.array(cumulated_true),np.array(cumulated_pred)))

def stats(true,pred):

	pred_2 = np.copy(pred)
	pred_2[pred<0.5] = 0
	pred_2[pred>=0.5] = 1

	try:
		roc = roc_auc_score(true,pred)
	except Exception as e:
		roc = 'N/A'
		print(e)
		
	acc = accuracy_score(true,pred_2)
	rec = recall_score(true,pred_2,pos_label=0)
	pre = precision_score(true,pred_2,pos_label=0)
	f1  = f1_score(true,pred_2,pos_label=0)

	string_result = ''

	string_result = '{0}{1}\n'.format(string_result,'Accuracy: {0}'.format(acc))
	string_result = '{0}{1}\n'.format(string_result,'AUC for ROC: {0}'.format(roc))
	string_result = '{0}{1}\n'.format(string_result,'Precision: {0}'.format(pre))
	string_result = '{0}{1}\n'.format(string_result,'Recall: {0}'.format(rec))
	string_result = '{0}{1}\n'.format(string_result,'F1-Score: {0}'.format(f1))
	
	return string_result

# mdl = fine_tuning_VGG16.load_full_model_outside_use(final_model_path)

# train_true,train_pred,val_true,val_pred = get_predictions(mdl)

# np.save('{0}/train_true_{1}.txt'.format(fixed_address,setting),train_true)
# np.save('{0}/train_pred_{1}.txt'.format(fixed_address,setting),train_pred)
# np.save('{0}/val_true_{1}.txt'.format(fixed_address,setting),val_true)
# np.save('{0}/val_pred_{1}.txt'.format(fixed_address,setting),val_pred)

# train_true=np.load('train_true_{0}.txt.npy'.format(setting))
# train_pred=np.load('train_pred_{0}.txt.npy'.format(setting))
# val_true=np.load('val_true_{0}.txt.npy'.format(setting))
# val_pred=np.load('val_pred_{0}.txt.npy'.format(setting))

# print(stats(val_true,val_pred))


# get_predictions_and_save('/space/ariyanzarei/test_patches','/space/ariyanzarei/results/2599/real_test',mdl)
# load_predictions_and_stats_test_patches('/space/ariyanzarei/test_patches','/space/ariyanzarei/results/2598/real_test')