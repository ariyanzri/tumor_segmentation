'''
This script contains preprocessing methods for the task of whole slide imaging.
'''

import cv2
import openslide
import numpy
import random
import annotation
import os
import time
import sys
import multiprocessing

'''
===================================== Bounding Box Approach =========================================
In this approach we use an image from a relatively high level of a wsi and use it for background 
subtraction. After subtracting background (thresholding, opening, closing and binarization), we 
find the boundry box that covers the whole foreground region in the middle of the slide. Then we 
draw random values of x,y inside this boundry box (in the scale that we want to generate patches) and
crop patches at the desired magnification level. One downside of this approach is that even in the 
foreground region, there might be still some empty spaces. therefore, we prefere the other approach.

=====================================================================================================

'''

'''
This function finds the smallest bounding box that covers all the foreground pixels in the center 
of the image. (we need to reverse the order of the x,y when we return, because of the wrong 
implementation of the for loops)

Inputs:
	img = the image to find its foreground bounding box

Output:
	y = the y coordinate of the upper left corner of the box 
	x = the x coordinate of the upper left corner of the box
	h = the height of the box
	w = the width of the box
'''
def crop_bg(img):
	x=y=w=h=0

	for i in range(0,numpy.shape(img)[0]):
		flag = False
		for j in range(0,numpy.shape(img)[1]):
			if img[i,j]>0:
				x=i
				flag = True
				break
		if flag:
			break

	for i in range(numpy.shape(img)[0]-1,-1,-1):
		flag = False
		for j in range(0,numpy.shape(img)[1]):
			if img[i,j]>0:
				w=i-x
				flag = True
				break
		if flag:
			break

	for j in range(0,numpy.shape(img)[1]):
		flag = False
		for i in range(0,numpy.shape(img)[0]):
			if img[i,j]>0:
				y=j
				flag = True
				break
		if flag:
			break

	for j in range(numpy.shape(img)[1]-1,-1,-1):
		flag = False
		for i in range(0,numpy.shape(img)[0]):
			if img[i,j]>0:
				h=j-y
				flag = True
				break
		if flag:
			break

	return y,x,h,w


'''
This function runs the preprocessing on an image of a slide at certain magnification level. This 
preprocessing includes converting image to HSV format, running Otsu threshoulding on each channel,
combining all three channels, runing closing and opening morphological operations in order to remove 
noises and at the end calling another function to get the bounding box that covers all the foreground
pixels.

Inputs:
	wsi = the whole slide image
	lvl = the level at wihch we want to subtract background

Output:
	lvl = the level at wihch we subtract the background
	y = the y coordinate of the upper left corner of the box 
	x = the x coordinate of the upper left corner of the box
	h = the height of the box
	w = the width of the box
'''
def background_subtraction_bounding_box(wsi,lvl):
	wsi = wsi.read_region((0,0),lvl,wsi.level_dimensions[lvl])
	wsi = numpy.array(wsi) 
	
	hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)
	
	h_channel = hsv[:,:,0]
	s_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]

	h_return,h_th = cv2.threshold(h_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	s_return,s_th = cv2.threshold(s_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	v_return,v_th = cv2.threshold(v_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	h_th[h_th==255] = 1
	s_th[s_th==255] = 1
	v_th[v_th==255] = 1

	bg = numpy.logical_or(h_th,s_th)
		
	bg=numpy.uint8(bg*255)

	kernel = numpy.ones((9,9),numpy.uint8)
	closing = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
	kernel = numpy.ones((35,35),numpy.uint8)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	x,y,h,w = crop_bg(opening)

	return lvl,x,y,w,h



def background_subtraction_contour(wsi,lvl):
	wsi = wsi.read_region((0,0),lvl,wsi.level_dimensions[lvl])
	wsi = numpy.array(wsi) 
	
	hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)
	
	h_channel = hsv[:,:,0]
	s_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]

	h_return,h_th = cv2.threshold(h_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	s_return,s_th = cv2.threshold(s_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	v_return,v_th = cv2.threshold(v_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# cv2.imshow('s',s_th)

	h_th[h_th==255] = 1
	s_th[s_th==255] = 1
	v_th[v_th==255] = 1

	bg = numpy.logical_and(s_th,s_th)
		
	bg=numpy.uint8(bg*255)

	kernel = numpy.ones((9,9),numpy.uint8)
	closing = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
	kernel = numpy.ones((9,9),numpy.uint8)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	# cv2.imshow('o',opening)
	image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	return contours,wsi


'''
This function uses the other two functions and generates patches. 

Inputs:
	wsi = the whole slide image
	patch_address = the address to which patch images would be written
	patch_width = the width of each patch
	patch_height = the height of each patch
	patch_level = the level at which we want to generate patches
	num_patches = the total number of patches that we want
	bg_level = the level at which we want to subtract the background

Output:

'''
def generate_patches_bounding_box(wsi,patch_address,patch_width,patch_height,patch_level,num_patches,bg_level):
	b_l,b_x,b_y,b_w,b_h = background_subtraction_bounding_box(wsi,bg_level)

	b_x = b_x*(2**(b_l-patch_level))
	b_y = b_y*(2**(b_l-patch_level))
	b_w = b_w*(2**(b_l-patch_level))
	b_h = b_h*(2**(b_l-patch_level))
	
	for i in range(0,num_patches):
		x = random.randint(b_x,b_x+b_w)
		y = random.randint(b_y,b_y+b_h)
		
		patch = numpy.array(wsi.read_region((x,y),patch_level,(patch_width,patch_height)))
		cv2.imwrite("{0}/patch-{1}.jpg".format(patch_address,i),patch)



'''
=================================== Foreground List Approach ========================================
In this approach we use an image from a relatively high level of a wsi and use it for background 
subtraction. After subtracting background (thresholding, opening, closing and binarization), we 
extract list of pixels covering the whole area of the foreground pixels. More specifically, the list
is a tuple of two lists, one for x and one for y. Using these lists, then we draw points from the 
foreground region and use them to generate patches. This approach seems to work better, since we don't
use empty regions that could exist in the bounding box in previous approach. 

=====================================================================================================

'''

'''
This function runs the preprocessing on an image of a slide at certain magnification level. This 
preprocessing includes converting image to HSV format, running Otsu threshoulding on each channel,
combining all three channels, runing closing and opening morphological operations in order to remove 
noises and at the end extracting list of foreground pixels. (tuple containing two lists)

Inputs:
	wsi = the whole slide image
	lvl = the level at wihch we want to subtract background
	show = whether to show the background images or not

Output:
	a tuple that consists of two list, first is the x coordinates and second is y's.
'''
def background_subtraction_foreground_list(wsi,lvl,show=False):

	wsi = wsi.read_region((0,0),lvl,wsi.level_dimensions[lvl])
	wsi = numpy.array(wsi) 
	
	hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)
	
	h_channel = hsv[:,:,0]
	s_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]

	h_return,h_th = cv2.threshold(h_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	s_return,s_th = cv2.threshold(s_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	v_return,v_th = cv2.threshold(v_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	h_th[h_th==255] = 1
	s_th[s_th==255] = 1
	v_th[v_th==255] = 1

	bg = numpy.logical_and(h_th,s_th)
		
	bg=numpy.uint8(bg*255)

	kernel = numpy.ones((5,5),numpy.uint8)
	closing = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
	kernel = numpy.ones((7,7),numpy.uint8)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	h_th[h_th==1] = 255
	s_th[s_th==1] = 255

	if show:
		cv2.imshow('Main image at level {0}'.format(lvl),wsi)
		cv2.imshow('H channel after otsu',h_th)
		cv2.imshow('S channel after otsu',s_th)
		cv2.imshow('Combined H and S',bg)
		cv2.imshow('Background after morphological',opening)

		cv2.waitKey(0)

	return numpy.nonzero(opening)


def background_subtraction_save(wsi,lvl,iid):

	wsi = wsi.read_region((0,0),lvl,wsi.level_dimensions[lvl])
	wsi = numpy.array(wsi) 
	
	hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)
	
	h_channel = hsv[:,:,0]
	s_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]

	h_return,h_th = cv2.threshold(h_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	s_return,s_th = cv2.threshold(s_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	v_return,v_th = cv2.threshold(v_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	h_th[h_th==255] = 1
	s_th[s_th==255] = 1
	v_th[v_th==255] = 1

	bg = numpy.logical_and(h_th,s_th)
		
	bg=numpy.uint8(bg*255)

	kernel = numpy.ones((5,5),numpy.uint8)
	closing = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
	kernel = numpy.ones((7,7),numpy.uint8)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2GRAY)

	res = numpy.concatenate((wsi,opening),axis=1)
	cv2.imwrite('/space/ariyanzarei/bgs/background_{0}.jpg'.format(iid),res)

'''
This function almost bounds the foreground list of pixels to the abnormal and metastatic areas. 

Inputs:
	l_x = the list of x coordinates of the forground in b_lbl level
	l_y = the list of y coordinates of the forground in b_lbl level
	xml_path = the path of the annotation file for the given wsi
	lvl = the patch level of magnification
	b_lvl = the magnification at which background subtraction happened

Outputs:
	the two list of x,y coordinates which were bounded between the min and max value of annotation
	areas.
'''
def get_bounded_list(l_x,l_y,xml_path,lvl,b_lvl):
	min_x,max_x,min_y,max_y = annotation.get_min_max_annotation(xml_path,lvl)
	
	min_x = min_x*(2**(lvl-b_lvl))
	max_x = max_x*(2**(lvl-b_lvl))
	min_y = min_y*(2**(lvl-b_lvl))
	max_y = max_y*(2**(lvl-b_lvl))

	x_list_1 = l_y>=min_x
	x_list_2 = l_y<=max_x
	x_list = numpy.logical_and(x_list_1,x_list_2)

	y_list_1 = l_x>=min_y
	y_list_2 = l_x<=max_y
	y_list = numpy.logical_and(y_list_1,y_list_2)

	final = numpy.logical_and(x_list,y_list)
	
	return l_x[final],l_y[final]


'''
This function converts the annotation into a mask and then return the list of all points inside

Inputs:
	root = the root object of the xml file
	level = the level of magnification that the mask is at

Output:
	the mask
'''
def get_annotated_list(xml,level):
	
	root = 	root = annotation.load_annotation_xml(xml)

	polygon_list = []

	list_x = []
	list_y = []

	for a in root.ASAP_Annotations.Annotations.Annotation:
		point_list = []

		for coordinate in a.Coordinates.Coordinate:
			point_list.append(annotation.Point(int(float(coordinate['X'])),int(float(coordinate['Y']))))

		polygon_list.append(point_list)

	min_x,max_x,min_y,max_y=annotation.get_min_max_annotation(xml,level)

	for i in range(min_x,max_x):
		for j in range(min_y,max_y):
			if annotation.is_inside_multi_poly(polygon_list,annotation.convert_magnification(annotation.Point(i,j),level,0)):
				list_x.append(i)
				list_y.append(j)

	# for i in range(0,200):
	# 	rnd = random.randint(0,np.shape(list_y)[0]-1)
	# 	x = list_y[rnd]
	# 	y = list_x[rnd]
	# 	mask[x][y] = 0

	# # mask = mask[~np.all(mask == 0, axis=0)]
	# mask = mask[~np.all(mask == 0, axis=1)]
	

	# # print(mask)
	# cv2.imshow('sdfs',mask)
	# cv2.waitKey(0)

	# return mask
	return list_x,list_y


'''
This function uses the other two functions and generates patches. 

Inputs:
	wsi = the whole slide image
	patch_address_normal = the address to which normal patch images would be written
	patch_address_abnormal = the address to which abnormal patch images would be written
	xml_path = the path for the xml annotation file
	patch_level = the level at which we want to generate patches
	num_patches = the total number of patches that we want
	bg_level = the level at which we want to subtract the background
	is_normal = determines whether the image is classified as normal or not (the whole slide image)
	det_thresh = the threshold for considering a patch abnormal (default = 0.3)
	patch_width = the width of each patch (default = 256)
	patch_height = the height of each patch (default = 256)
	

Output:

'''
def generate_patches_foreground_list(wsi,patch_address_normal,patch_address_abnormal,xml_path,patch_level,num_patches,bg_level,is_normal,det_thresh=0.3,patch_width=256,patch_height=256):
	fg_lists = background_subtraction_foreground_list(wsi,bg_level)

	list_x = fg_lists[0]
	list_y = fg_lists[1]
	
	# for abnormal wsis, try to generate patches inside the metastatic area
	if not is_normal:
		# list_x,list_y = get_bounded_list(list_x,list_y,xml_path,patch_level,bg_level)
		list_x,list_y = get_annotated_list(xml_path,bg_level)

	i_normal = len(os.listdir(patch_address_normal))
	i_abnormal = len(os.listdir(patch_address_abnormal))

	for i in range(0,num_patches):
		rnd = random.randint(0,numpy.shape(list_y)[0]-1)
		x = list_y[rnd]*(2**(bg_level-patch_level))
		y = list_x[rnd]*(2**(bg_level-patch_level))

		patch = numpy.array(wsi.read_region((x-patch_width/2,y-patch_height/2),patch_level,(patch_width,patch_height)))
		center = annotation.Point(0,0)
		location = annotation.Point(int(x-patch_width/2),int(y-patch_height/2))
		
		if is_normal or not annotation.is_abnormal(wsi,xml_path,patch_level,center,location,False,det_thresh,patch_width,patch_height):
			cv2.imwrite("{0}/patch-{1}.jpg".format(patch_address_normal,i_normal),patch)
			i_normal+=1
		else:
			cv2.imwrite("{0}/patch-{1}.jpg".format(patch_address_abnormal,i_abnormal),patch)
			i_abnormal+=1
		
		#sys.stdout.write("\r"+'Number of generated Patches: {0} normal, {1} abnormal\r'.format(i_normal,i_abnormal))
		#sys.stdout.flush()
		#time.sleep(0.0001)
	#print


'''
This function is the same as before with slight changes for multiprocessing use. 

Inputs:
	wsi = the whole slide image
	patch_address_normal = the address to which normal patch images would be written
	patch_address_abnormal = the address to which abnormal patch images would be written
	xml_path = the path for the xml annotation file
	patch_level = the level at which we want to generate patches
	num_patches = the total number of patches that we want
	bg_level = the level at which we want to subtract the background
	is_normal = determines whether the image is classified as normal or not (the whole slide image)
	det_thresh = the threshold for considering a patch abnormal (default = 0.3)
	patch_width = the width of each patch (default = 256)
	patch_height = the height of each patch (default = 256)
	

Output:

'''
def generate_patches_foreground_list_multi_process(wsi_address,patch_address_normal,patch_address_abnormal,xml_path,patch_level,num_patches,bg_level,is_normal,wsi_id,det_thresh=0.3,patch_width=256,patch_height=256):
	wsi = openslide.OpenSlide(wsi_address)
	
	if wsi.level_count-1<bg_level:
		bg_level = wsi.level_count -1

	fg_lists = background_subtraction_foreground_list(wsi,bg_level)

	list_x = fg_lists[0]
	list_y = fg_lists[1]
	
	# for abnormal wsis, try to generate patches inside the metastatic area
	if not is_normal:
		# list_x,list_y = get_bounded_list(list_x,list_y,xml_path,patch_level,bg_level)
		list_x,list_y = get_annotated_list(xml_path,bg_level)
	else:
		tmp = list_x
		list_x = list_y
		list_y = tmp

	i_normal = 0
	i_abnormal = 0

	if numpy.shape(list_y)[0] == 0:
		print('annotation problem for slide {0}. annotated area is empty'.format(wsi_id))
		return

	for i in range(0,num_patches):
		rnd = random.randint(0,numpy.shape(list_y)[0]-1)
		x = list_x[rnd]*(2**(bg_level-patch_level))
		y = list_y[rnd]*(2**(bg_level-patch_level))

		patch = numpy.array(wsi.read_region((x-int(patch_width/2),y-int(patch_height/2)),patch_level,(patch_width,patch_height)))
		center = annotation.Point(0,0)
		location = annotation.Point(int(x-patch_width/2),int(y-patch_height/2))
		
		if is_normal or not annotation.is_abnormal(wsi,xml_path,patch_level,center,location,False,det_thresh,patch_width,patch_height):
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address_normal,wsi_id,i_normal),patch)
			i_normal+=1
		else:
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address_abnormal,wsi_id,i_abnormal),patch)
			i_abnormal+=1

	print('patch generation finished successfully for {0}.'.format(wsi_id))
	

'''
This function is the same as before with slight fixes for the duplicate patch generation problem 

Inputs:
	wsi = the whole slide image
	patch_address_normal = the address to which normal patch images would be written
	patch_address_abnormal = the address to which abnormal patch images would be written
	xml_path = the path for the xml annotation file
	patch_level = the level at which we want to generate patches
	num_patches = the total number of patches that we want
	bg_level = the level at which we want to subtract the background
	is_normal = determines whether the image is classified as normal or not (the whole slide image)
	det_thresh = the threshold for considering a patch abnormal (default = 0.3)
	patch_width = the width of each patch (default = 256)
	patch_height = the height of each patch (default = 256)
	

Output:

'''
def generate_patches_foreground_list_multi_process_no_duplicates(wsi_address,patch_address_normal,patch_address_abnormal,xml_path,patch_level,num_patches,bg_level,is_normal,wsi_id,det_thresh=0.3,patch_width=256,patch_height=256):
	wsi = openslide.OpenSlide(wsi_address)
	
	if wsi.level_count-1<bg_level:
		bg_level = wsi.level_count -1

	fg_lists = background_subtraction_foreground_list(wsi,bg_level)

	list_x = fg_lists[0]
	list_y = fg_lists[1]
	
	# for abnormal wsis, try to generate patches inside the metastatic area
	if not is_normal:
		# list_x,list_y = get_bounded_list(list_x,list_y,xml_path,patch_level,bg_level)
		list_x,list_y = get_annotated_list(xml_path,bg_level)
	else:
		tmp = list_x
		list_x = list_y
		list_y = tmp

	i_normal = 0
	i_abnormal = 0

	if numpy.shape(list_y)[0] == 0:
		print('annotation problem for slide {0}. annotated area is empty'.format(wsi_id))
		return

	sample_count = min(num_patches,numpy.shape(list_y)[0])
	if sample_count != num_patches:
		print('   -- generating fewer patches than requested number for this slide.')

	samples_indices = random.sample(range(0,numpy.shape(list_y)[0]),sample_count)
	
	for i in range(0,sample_count):
		sample_index = samples_indices[i]
		x = list_x[sample_index]*(2**(bg_level-patch_level))
		y = list_y[sample_index]*(2**(bg_level-patch_level))

		patch = numpy.array(wsi.read_region((x-int(patch_width/2),y-int(patch_height/2)),patch_level,(patch_width,patch_height)))
		center = annotation.Point(0,0)
		location = annotation.Point(int(x-patch_width/2),int(y-patch_height/2))
		
		if is_normal or not annotation.is_abnormal(wsi,xml_path,patch_level,center,location,False,det_thresh,patch_width,patch_height):
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address_normal,wsi_id,i_normal),patch)
			i_normal+=1
		else:
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address_abnormal,wsi_id,i_abnormal),patch)
			i_abnormal+=1


	print('Generated {0} patches for {1}.'.format(sample_count, wsi_id))
	

'''
This function is the same as before with slight changes for multiprocessing use. It also saves the coordinates of the centers of the patches
alongside with the percentage of the abnormal area. It is intended for testing purposes.  

Inputs:
	wsi = the whole slide image
	patch_address = the address to which patch images would be written
	xml_path = the path for the xml annotation file
	patch_level = the level at which we want to generate patches
	num_patches = the total number of patches that we want
	bg_level = the level at which we want to subtract the background
	is_normal = determines whether the image is classified as normal or not (the whole slide image)
	det_thresh = the threshold for considering a patch abnormal (default = 0.3)
	patch_width = the width of each patch (default = 256)
	patch_height = the height of each patch (default = 256)
	

Output:

'''
def generate_patches_foreground_list_multi_process_for_test(wsi_address,patch_address,xml_path,patch_level,num_patches,bg_level,is_normal,wsi_id,det_thresh=0.3,patch_width=256,patch_height=256):
	wsi = openslide.OpenSlide(wsi_address)
	
	if wsi.level_count-1<bg_level:
		bg_level = wsi.level_count -1

	fg_lists = background_subtraction_foreground_list(wsi,bg_level)

	list_x = fg_lists[0]
	list_y = fg_lists[1]
	
	# for abnormal wsis, try to generate patches inside the metastatic area
	if not is_normal:
		# list_x,list_y = get_bounded_list(list_x,list_y,xml_path,patch_level,bg_level)
		list_x,list_y = get_annotated_list(xml_path,bg_level)
	else:
		tmp = list_x
		list_x = list_y
		list_y = tmp

	i_normal = 0
	i_abnormal = 0

	if numpy.shape(list_y)[0] == 0:
		print('annotation problem for slide {0}. annotated area is empty'.format(wsi_id))
		return

	for i in range(0,num_patches):
		rnd = random.randint(0,numpy.shape(list_y)[0]-1)
		x = list_x[rnd]*(2**(bg_level-patch_level))
		y = list_y[rnd]*(2**(bg_level-patch_level))

		patch = numpy.array(wsi.read_region((x-int(patch_width/2),y-int(patch_height/2)),patch_level,(patch_width,patch_height)))
		center = annotation.Point(0,0)
		location = annotation.Point(int(x-patch_width/2),int(y-patch_height/2))
		
		if is_normal:
			cv2.imwrite("{0}/n_{1}_{2}_{3}-{4}_{5}.jpg".format(patch_address,wsi_id,i_normal,x,y,0),patch)
			i_normal+=1
		else:
			perc_abnormal = annotation.abnormal_perc(wsi,xml_path,patch_level,center,location,False,patch_width,patch_height)
			
			cv2.imwrite("{0}/a_{1}_{2}_{3}-{4}_{5}.jpg".format(patch_address,wsi_id,i_abnormal,x,y,perc_abnormal),patch)
			i_abnormal+=1
		
	print('patch generation finished successfully for {0}.'.format(wsi_id))
	

'''
This function generate patches for given wsi images. 

Input:
	slide_data_folder = the location of the slide_data train folder
	patch_data_folder = the location of the patch_data train folder
	annotation_folder = the locaiton of the annotation folder
	per_wsi_patch_no = the number of patches to be generated for each wsi
	patch_level = the magnification level at which patches will be generated
	bg_level = the magnification level at which the background subtraction algorithm works with wsi
Output:
	nothing 
'''
def generate_patches_batch(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):

	for filename in os.listdir(slide_data_folder+'/normal'):
		wsi = openslide.OpenSlide(slide_data_folder+'/normal/'+filename)
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		generate_patches_foreground_list(wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,True)

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		wsi = openslide.OpenSlide(slide_data_folder+'/abnormal/'+filename)
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		generate_patches_foreground_list(wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,False)



'''
This function is the same as before with slight changes for Multiprocessing. 

Input:
	slide_data_folder = the location of the slide_data train folder
	patch_data_folder = the location of the patch_data train folder
	annotation_folder = the locaiton of the annotation folder
	per_wsi_patch_no = the number of patches to be generated for each wsi
	patch_level = the magnification level at which patches will be generated
	bg_level = the magnification level at which the background subtraction algorithm works with wsi
Output:
	nothing 
'''
def generate_patches_batch_multi_process(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	
	processes = multiprocessing.Pool(35)
	
	list_args = []

	for filename in os.listdir(slide_data_folder+'/normal'):
		wsi = slide_data_folder+'/normal/'+filename
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
		list_args.append((wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif','')))
		

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		wsi = slide_data_folder+'/abnormal/'+filename
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
		list_args.append((wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))
		
	processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args)

'''
This function is the same as the one above, but it genertes the patches in each test, training and validation separately (the second approach)

Input:
	slide_data_folder = the location of the slide_data train folder
	patch_data_folder = the location of the patch_data train folder
	annotation_folder = the locaiton of the annotation folder
	per_wsi_patch_no = the number of patches to be generated for each wsi
	patch_level = the magnification level at which patches will be generated
	bg_level = the magnification level at which the background subtraction algorithm works with wsi
Output:
	nothing 
'''
def generate_patches_batch_multi_process_separate_first(slide_data_folder,training_patch_data_folder,validation_patch_data_folder, \
	test_patch_data_folder,training_slide_names,validation_slide_names,test_slide_names,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	

	for filename in os.listdir(slide_data_folder+'/normal'):
		wsi = openslide.OpenSlide(slide_data_folder+'/normal/'+filename)
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		# generate_patches_foreground_list_multi_process(wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif',''))
		
		if filename in training_slide_names:
			patch_data_folder = training_patch_data_folder
		elif filename in validation_slide_names:
			patch_data_folder = validation_patch_data_folder
		elif filename in test_slide_names:
			patch_data_folder = test_patch_data_folder 

		p = multiprocessing.Process(target=generate_patches_foreground_list_multi_process,args=(wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif',''),))
		p.start()


	for filename in os.listdir(slide_data_folder+'/abnormal'):
		wsi = openslide.OpenSlide(slide_data_folder+'/abnormal/'+filename)
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		

		if filename in training_slide_names:
			patch_data_folder = training_patch_data_folder
		elif filename in validation_slide_names:
			patch_data_folder = validation_patch_data_folder
		elif filename in test_slide_names:
			patch_data_folder = test_patch_data_folder


		p = multiprocessing.Process(target=generate_patches_foreground_list_multi_process,args=(wsi,patch_data_folder+'/normal',patch_data_folder+'/abnormal',annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif',''),))
		p.start()


'''
This function is a helper auxilary function for doing the multiprocessing with pool

'''
def generate_patches_foreground_list_multi_process_pool_helper(args):
	# generate_patches_foreground_list_multi_process(*args)
	generate_patches_foreground_list_multi_process_no_duplicates(*args)



'''
This function is a helper auxilary function for doing the multiprocessing with pool

'''
def generate_patches_foreground_list_multi_process_for_test_pool_helper(args):
	generate_patches_foreground_list_multi_process_for_test(*args)


'''
This function generates patches for each slide separately. 

Input:
	slide_data_folder = the location of the slide_data train folder
	patch_data_folder = the location of the patch_data train folder
	annotation_folder = the locaiton of the annotation folder
	per_wsi_patch_no = the number of patches to be generated for each wsi
	patch_level = the magnification level at which patches will be generated
	bg_level = the magnification level at which the background subtraction algorithm works with wsi
Output:
	nothing 
'''
def generate_patches_batch_multi_process_byslide(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	

	# for filename in os.listdir(slide_data_folder+'/normal'):
	# 	wsi = openslide.OpenSlide(slide_data_folder+'/normal/'+filename)
	# 	annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

	# 	p = multiprocessing.Process(target=generate_patches_foreground_list_multi_process,args=(wsi,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
	# 		,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif',''),))
	# 	p.start()


	# for filename in os.listdir(slide_data_folder+'/abnormal'):
	# 	wsi = openslide.OpenSlide(slide_data_folder+'/abnormal/'+filename)
	# 	annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		

	# 	p = multiprocessing.Process(target=generate_patches_foreground_list_multi_process,args=(wsi,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
	# 		,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif',''),))
	# 	p.start()

	#  THE POOL -----------------------------------------------------------

	# normal_processes = multiprocessing.Pool(48)
	# abnormal_processes = multiprocessing.Pool(48)
	
	# list_args_normal = []
	# list_args_abnormal = []


	# for filename in os.listdir(slide_data_folder+'/normal'):
	# 	annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

	# 	list_args_normal.append((slide_data_folder+'/normal/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
	# 		,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif','')))
		

	# for filename in os.listdir(slide_data_folder+'/abnormal'):
	# 	annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
	# 	list_args_abnormal.append((slide_data_folder+'/abnormal/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
	# 		,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))

	# normal_processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args_normal)
	# abnormal_processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args_abnormal)

	#  THE POOL With fixed multiprocessing problem-----------------------------------------------------------

	processes = multiprocessing.Pool(35)
	
	list_args = []


	for filename in os.listdir(slide_data_folder+'/normal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

		list_args.append((slide_data_folder+'/normal/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
			,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif','')))
		

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
		list_args.append((slide_data_folder+'/abnormal/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
			,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))

	processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args)

	# __________________________ MISSINGS _______________________________

	# list_missing = ['tumor_002','tumor_004','tumor_008','tumor_009','tumor_010','tumor_012','tumor_014','tumor_017','tumor_023','tumor_024','tumor_028','tumor_030','tumor_035','tumor_038','tumor_040','tumor_043','tumor_049','tumor_053','tumor_054','tumor_057','tumor_059','tumor_060','tumor_063','tumor_065','tumor_066','tumor_067','tumor_068','tumor_070','tumor_074','tumor_077','tumor_081','tumor_086','tumor_095','tumor_097','tumor_098','tumor_100']

	# processes = multiprocessing.Pool(36)
	
	# list_args = []

	# for f in list_missing:
	# 	filename = '{0}.tif'.format(f)

	# 	annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
	# 	list_args.append((slide_data_folder+'/abnormal/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
	# 		,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))

	# processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args)


def generate_patches_batch_multi_process_byslide_for_test(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	
	normal_processes = multiprocessing.Pool(48)
	abnormal_processes = multiprocessing.Pool(48)
	
	list_args_normal = []
	list_args_abnormal = []


	for filename in os.listdir(slide_data_folder+'/normal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

		list_args_normal.append((slide_data_folder+'/normal/'+filename,patch_data_folder+'/{0}'.format(filename.replace('.tif',''))\
			,annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif','')))
		

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
		list_args_abnormal.append((slide_data_folder+'/abnormal/'+filename,patch_data_folder+'/{0}'.format(filename.replace('.tif',''))\
			,annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))

	normal_processes.map(generate_patches_foreground_list_multi_process_for_test_pool_helper,list_args_normal)
	abnormal_processes.map(generate_patches_foreground_list_multi_process_for_test_pool_helper,list_args_abnormal)



def generate_patch_directories_for_slides(slide_data_folder,patch_data_folder):

	for filename in os.listdir(slide_data_folder+'/normal'):
		os.mkdir('{0}/{1}'.format(patch_data_folder,filename.replace('.tif','')))
		os.mkdir('{0}/{1}/normal'.format(patch_data_folder,filename.replace('.tif','')))
		os.mkdir('{0}/{1}/abnormal'.format(patch_data_folder,filename.replace('.tif','')))

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		os.mkdir('{0}/{1}'.format(patch_data_folder,filename.replace('.tif','')))
		os.mkdir('{0}/{1}/normal'.format(patch_data_folder,filename.replace('.tif','')))
		os.mkdir('{0}/{1}/abnormal'.format(patch_data_folder,filename.replace('.tif','')))


def generate_patch_directories_for_slides_test(slide_data_folder,patch_data_folder):

	for filename in os.listdir(slide_data_folder+'/normal'):
		os.mkdir('{0}/{1}'.format(patch_data_folder,filename.replace('.tif','')))
		
	for filename in os.listdir(slide_data_folder+'/abnormal'):
		os.mkdir('{0}/{1}'.format(patch_data_folder,filename.replace('.tif','')))
		

# code for new experiment regarding the comparison of the two kinds of normal patches.

def generate_patch_both_normal_new_method(wsi_address,patch_address,xml_path,patch_level,num_patches,bg_level,is_normal,wsi_id,det_thresh=0.001,patch_width=256,patch_height=256):
	wsi = openslide.OpenSlide(wsi_address)
	
	if wsi.level_count-1<bg_level:
		bg_level = wsi.level_count -1

	fg_lists = background_subtraction_foreground_list(wsi,bg_level)

	list_x = fg_lists[0]
	list_y = fg_lists[1]

	if numpy.shape(list_y)[0] == 0:
		print('annotation problem for slide {0}. annotated area is empty'.format(wsi_id))
		return

	sample_count = min(num_patches,numpy.shape(list_y)[0])
	if sample_count != num_patches:
		print('   -- generating fewer patches than requested number for this slide.')

	samples_indices = random.sample(range(0,numpy.shape(list_y)[0]),sample_count)
	
	for i in range(0,sample_count):
		sample_index = samples_indices[i]
		x = list_x[sample_index]*(2**(bg_level-patch_level))
		y = list_y[sample_index]*(2**(bg_level-patch_level))

		patch = numpy.array(wsi.read_region((x-int(patch_width/2),y-int(patch_height/2)),patch_level,(patch_width,patch_height)))
		center = annotation.Point(0,0)
		location = annotation.Point(int(x-patch_width/2),int(y-patch_height/2))
		
		if is_normal:
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address,wsi_id,i),patch)
		elif not annotation.is_abnormal(wsi,xml_path,patch_level,center,location,False,det_thresh,patch_width,patch_height):
			cv2.imwrite("{0}/{1}-patch_{2}.jpg".format(patch_address,wsi_id,i),patch)

	print('Generated {0} patches for {1}.'.format(sample_count, wsi_id))

def generate_patch_both_normal_new_method_pool_helper(args):
	generate_patch_both_normal_new_method(*args)

def generate_patch_both_normal_new_method_by_slide_parallel(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	
	processes = multiprocessing.Pool(35)
	
	list_args = []

	for filename in os.listdir(slide_data_folder+'/normal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

		list_args.append((slide_data_folder+'/normal/'+filename,patch_data_folder+'/real_normal'\
			,annotation_path,patch_level,per_wsi_patch_no,bg_level,True,filename.replace('.tif','')))
		

	for filename in os.listdir(slide_data_folder+'/abnormal'):
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')
		
		list_args.append((slide_data_folder+'/abnormal/'+filename,patch_data_folder+'/cancerous_normal'\
			,annotation_path,patch_level,per_wsi_patch_no,bg_level,False,filename.replace('.tif','')))

	processes.map(generate_patch_both_normal_new_method_pool_helper,list_args)


def generate_patches_batch_multi_process_byslide_for_test_slides(slide_data_folder,patch_data_folder,annotation_folder,per_wsi_patch_no,patch_level,bg_level):
	
	# for filename in os.listdir(slide_data_folder):
	# 	os.mkdir('{0}/{1}'.format(patch_data_folder,filename.replace('.tif','')))
	# 	os.mkdir('{0}/{1}/normal'.format(patch_data_folder,filename.replace('.tif','')))
	# 	os.mkdir('{0}/{1}/abnormal'.format(patch_data_folder,filename.replace('.tif','')))

	processes = multiprocessing.Pool(2)
	
	list_args = []
	# os.listdir(slide_data_folder)
	for filename in ['test_021.tif','test_090.tif']:
		annotation_path = annotation_folder+'/'+filename.replace('.tif','.xml')

		if os.path.exists(annotation_path):
			is_normal = False
		else:
			is_normal = True

		list_args.append((slide_data_folder+'/'+filename,patch_data_folder+'/{0}/normal'.format(filename.replace('.tif',''))\
			,patch_data_folder+'/{0}/abnormal'.format(filename.replace('.tif','')),annotation_path,patch_level,per_wsi_patch_no,bg_level,is_normal,filename.replace('.tif','')))
		
	processes.map(generate_patches_foreground_list_multi_process_pool_helper,list_args)


# ===================================================================================================


patch_no_wsi = 2000

# generate_patches_foreground_list_multi_process_no_duplicates('/data/histology/data/complete_training_slides/abnormal/tumor_005.tif',\
# 			'/space/ariyanzarei/tmp/normal','/space/ariyanzarei/tmp/abnormal','/data/histology/data/complete_training_slides/lesion_annotations/tumor_005.xml'\
# 			,0,patch_no_wsi,4,False,'tumor_005')

# generate_patch_directories_for_slides_test('/data/histology/data/complete_training_slides','/space/ariyanzarei/complete_dataset_test')
# print('folders created....')
# generate_patches_batch_multi_process_byslide_for_test('/data/histology/data/complete_training_slides','/space/ariyanzarei/complete_dataset_test','/data/histology/data/complete_training_slides/lesion_annotations',patch_no_wsi,0,8)
# print('patches generated....')

# generate_patches_batch('/data/histology/data/10p_training_slides','/space/ariyanzarei/generated_patches','/data/histology/data/10p_training_slides/lesion_annotations',patch_no_wsi,0,8)

# Multiprocess version
# generate_patches_batch_multi_process('/data/histology/data/10p_training_slides','/space/ariyanzarei/generated_patches','/data/histology/data/10p_training_slides/lesion_annotations',patch_no_wsi,0,8)
# generate_patches_batch_multi_process('/home/ariyan/IVILAB/Pathology/Codes/data/slide_data/train','/home/ariyan/IVILAB/Pathology/Codes/data/test_data','/home/ariyan/IVILAB/Pathology/Codes/data/slide_data/train/lesion_annotations',patch_no_wsi,0,8)

# Multiprocess and Separate First 
# tr_list = ['normal_099.tif', 'normal_033.tif', 'normal_056.tif', 'normal_091.tif', 'normal_118.tif', 'normal_036.tif', 'normal_031.tif' ,\
#  'normal_001.tif', 'normal_153.tif', 'tumor_076.tif', 'tumor_104.tif', 'tumor_108.tif', 'tumor_018.tif', 'tumor_013.tif', 'tumor_065.tif']
# va_list = ['normal_096.tif','normal_011.tif','normal_119.tif','tumor_016.tif','tumor_103.tif']
# te_list = ['normal_122.tif','normal_156.tif','normal_094.tif', 'normal_146.tif','tumor_011.tif','tumor_092.tif','tumor_023.tif']

# generate_patches_batch_multi_process_separate_first('/data/histology/data/10p_training_slides','/space/ariyanzarei/10p_dataset_55_20_25/training',\
# 	'/space/ariyanzarei/10p_dataset_55_20_25/validation','/space/ariyanzarei/10p_dataset_55_20_25/test',tr_list,va_list,te_list,\
# 	'/data/histology/data/10p_training_slides/lesion_annotations',patch_no_wsi,0,8)
# generate_patches_foreground_list_multi_process(openslide.OpenSlide('/home/ariyan/IVILAB/Pathology/Codes/data/slide_data/train/normal/normal_001.tif'),'/home/ariyan/Desktop/n','/home/ariyan/Desktop/a','/home/ariyan/IVILAB/Pathology/Codes/data/slide_data/train/lesion_annotations/tumor_001.xml',0,20,8,True,1)


# generate_patches_batch_multi_process_byslide('/data/histology/data/20p_training_slides','/space/ariyanzarei/20p_dataset',\
# 	'/data/histology/data/10p_training_slides/lesion_annotations',patch_no_wsi,0,8)

# original = sys.stdout
# # sys.stdout = open('log_generate_20p_for_hyper.txt', 'w')

# # generate_patch_directories_for_slides('/data/histology/data/20p_training_slides','/space/ariyanzarei/20p_hyper_dataset')

# # generate_patches_batch_multi_process_byslide('/data/histology/data/20p_training_slides','/space/ariyanzarei/20p_hyper_dataset',\
# # 	'/data/histology/data/20p_training_slides/lesion_annotations',patch_no_wsi,0,5)

# sys.stdout = open('log_generate_test.txt', 'w')

# generate_patches_batch_multi_process_byslide_for_test_slides('/data/histology/data/test_slides','/space/ariyanzarei/test_patches','/data/histology/data/test_lesion',2000,0,5)

# sys.stdout = original

# slide_data_folder = '/data/histology/data/complete_training_slides'

# for filename in os.listdir(slide_data_folder+'/normal'):
# 	wsi = openslide.OpenSlide(slide_data_folder+'/normal/'+filename)
# 	bg_level = 8
# 	if wsi.level_count-1<bg_level:
# 		bg_level = wsi.level_count -1

# 	background_subtraction_save(wsi,bg_level,filename)

# for filename in os.listdir(slide_data_folder+'/abnormal'):
# 	wsi = openslide.OpenSlide(slide_data_folder+'/abnormal/'+filename)
# 	bg_level = 8
# 	if wsi.level_count-1<bg_level:
# 		bg_level = wsi.level_count -1

# 	background_subtraction_save(wsi,bg_level,filename)