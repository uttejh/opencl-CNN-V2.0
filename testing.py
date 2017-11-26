import numpy
from PIL import Image
import Image
from procedures import *
import time
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=numpy.nan)
# numOfFiltersLayer1 = 20
# numOfFiltersLayer2 = 40

# numOfInputs1 = numOfFiltersLayer1*28*28		
# numOfOutputs1 = numOfFiltersLayer1*28*28	

# numOfInputs2 = numOfFiltersLayer1*numOfFiltersLayer2*14*14		
# numOfOutputs2 = numOfFiltersLayer1*numOfFiltersLayer2*14*14		

f = open('./weights/b1.txt')
b1 = pickle.load(f)
f.close()

f = open('./weights/b2.txt')
b2 = pickle.load(f)
f.close()

f = open('./weights/bFC.txt')
bFC = pickle.load(f)
f.close()

f = open('./weights/bhl.txt')
bhl = pickle.load(f)
f.close()

f = open('./weights/filters1.txt')
filters1 = pickle.load(f)
f.close()

f = open('./weights/filters2.txt')
filters2 = pickle.load(f)
f.close()

f = open('./weights/FC_to_HL.txt')
weights_FC_to_HL = pickle.load(f)
f.close()

f = open('./weights/HL_to_output.txt')
weights_HL_to_output = pickle.load(f)
f.close()

# print filters1[0]
# print filters2[0]
# print b1
# print b2
# print bhl
# print bFC
# print weights_HL_to_output[0]
# print weights_FC_to_HL[0]





# # alpha = 0.1

# # epochs = 2000

def readAllImages():
	data = []
	for i in range(1):
		# name = './dataset/'+str(i)+'.jpg'
		name = './test/img_115.jpg'
		image = Image.open(name)

		im = numpy.array( image, dtype="double" ) 
		data.append(im)

	return data

imagedata = readAllImages()

# # Reads image and converts it to an array - our input
# def readImage(x):
# 	# Open Image
# 	image = Image.open('1.jpg')

# 	# Resize each image to one desired shape
# 	# image = image.resize((28,28),Image.ANTIALIAS)

# 	# # Save as ...., "optimize" - reduce size (bytes)
# 	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

# 	data = numpy.array( image, dtype="double" ) 
# 	return data


p = Procedures()

totalloss=[]

# if iterat_epoch%100 == 0:
# 	print '###############################################'
# 	print 'Output at epoch '+str(iterat_epoch)+' is:'
# 	print '###############################################'

for iterat_image in range(1):
	# -------------------------------------- READ INPUT -------------------------------------------

	input_data = imagedata[iterat_image]
	# read labels
	# code this later
	# label = read_from....(label_list.txt)
	
	# Get shape and size/order
	input_shape1 = input_data.shape
	numinputs_pad1 = 1
	order_pad1 = input_shape1[0]

	# tt = time.clock() - start
	# print "Reading image: " + str(tt)

	# -----------------------------------------------------------------------------------------------
	#                     [ PADDING --> CONVOLUTION ] --> RELU --> POOLING (FIRST ITERATION)
	# -----------------------------------------------------------------------------------------------


	# -----------------------------------------PADDING-----------------------------------------------

	pad1 = p.pad(input_data, numinputs_pad1, order_pad1)
	pad_arr1 = array(pad1)
	pad_shape1 = pad_arr1.shape

	# tt = time.clock() - tt
	# print "Padding: " + str(tt)

	# ---------------------------------------CONVOLUTION---------------------------------------------

	numinputs_conv1 = 1
	order_conv1 = pad_shape1[1]

	# Convolute with input and mention no. of filters to be used
	conv_layer1 = p.convolution(pad1[0], filters1, b1, numinputs_conv1, order_conv1)	
	conv1_shape = array(conv_layer1).shape
	
	# tt = time.clock() - tt
	# print "CONVOLUTION: " + str(tt)

	# ------------------------------------------RELU--------------------------------------------------

	numinputs_relu1 = conv1_shape[0]
	order_relu1 = conv1_shape[1]

	relu1 = p.relu(conv_layer1, numinputs_relu1, order_relu1)

	relu_arr1 = array(relu1)
	relu1_shape = relu_arr1.shape

	# tt = time.clock() - tt
	# print "RELU: " + str(tt)

	# -----------------------------------------POOLING-------------------------------------------------

	numinputs_pool1 = relu1_shape[0]
	order_pool1 = relu1_shape[1]
	
	pool1,index1 = p.pooling(relu1, numinputs_pool1, order_pool1)

	pool_arr1 = array(pool1)
	pool1_shape = pool_arr1.shape

	# tt = time.clock() - tt
	# print "POOLING: " + str(tt)

	# x = time.clock() - start
	# print "One iter time :" + str(x)

	# -----------------------------------------------------------------------------------------------
	#                     [ PADDING --> CONVOLUTION ] --> RELU --> POOLING (SECOND ITERATION)
	# -----------------------------------------------------------------------------------------------


	# -----------------------------------------PADDING-----------------------------------------------

	numinputs_pad2 = pool1_shape[0]
	order_pad2 = pool1_shape[1]

	pad2 = p.pad(pool1, numinputs_pad2, order_pad2)

	pad2_shape = array(pad2).shape

	# tt = time.clock() - tt
	# print "Padding: " + str(tt)
	
	# ---------------------------------------CONVOLUTION---------------------------------------------

	numinputs_conv2 = pad2_shape[0]
	order_conv2 = pad2_shape[1]

	# Convolute with input and mention no. of filters to be used
	conv_layer2 = p.convolution(pad2, filters2, b2, numinputs_conv2, order_conv2)	
	conv2_shape = array(conv_layer2).shape

	# tt = time.clock() - tt
	# print "CONVOLUTION: " + str(tt)

	# ------------------------------------------RELU--------------------------------------------------

	numinputs_relu2 = conv2_shape[0]
	order_relu2 = conv2_shape[1]

	relu2 = p.relu(conv_layer2, numinputs_relu2, order_relu2)

	relu_arr2 = array(relu2)
	relu2_shape = relu_arr2.shape		

	# tt = time.clock() - tt
	# print "RELU: " + str(tt)
	# -----------------------------------------POOLING-------------------------------------------------

	numinputs_pool2 = relu2_shape[0]
	order_pool2 = relu2_shape[1]
	
	pool2,index2 = p.pooling(relu2, numinputs_pool2, order_pool2)

	pool_arr2 = array(pool2)
	pool2_shape = pool_arr2.shape

	# tt = time.clock() - tt
	# print "POOLING: " + str(tt)

	# x = time.clock() - x
	# print "Second iter time :" + str(x)
	# ---------------------------------- END OF SECOND ITERATION ---------------------------------------


	# --------------------------------------------------------------------------------------------------
	# ------------------------------[ FC --> HIDDEN LAYER --> OUTPUT ]----------------------------------
	# --------------------------------------------------------------------------------------------------


	# ----------------------------------- FULLY CONNECTED LAYER ----------------------------------------

	FC = array(pool2).ravel()

	# tt = time.clock() - tt
	# print "FC: " + str(tt)
	# ------------------------------------ FC --> Hidden Layer -----------------------------------------

	# Used numpy functions :- since the given input is smaller which therefore takes less
	# time than a GPU function
	# if iterat_epoch == 0 and iterat_image == 0:
	# 	numOfHiddenNeurons = 100
	# 	numOfOutputNeurons = 10

	# 	n_in1 = FC.shape[0]
	# 	n_out1 = numOfHiddenNeurons
	# 	w_bound1 = numpy.sqrt(6./float(n_in1+n_out1))
	# 	weights_FC_to_HL = numpy.random.uniform(-w_bound1,w_bound1,(numOfHiddenNeurons, n_in1))

	# 	n_in2 = numOfHiddenNeurons
	# 	n_out2 = numOfOutputNeurons
	# 	w_bound2 = numpy.sqrt(6./float(n_in2+n_out2))
	# 	weights_HL_to_output = numpy.random.uniform(-w_bound2,w_bound2,(numOfOutputNeurons, numOfHiddenNeurons))

	HL_WX_plus_b = numpy.dot(weights_FC_to_HL, FC) + bFC

	minmax_scale = MinMaxScaler(feature_range=(-1, 1), copy=True)
	# normalize
	HL_WX_plus_b_shape = array(HL_WX_plus_b).shape
	HL_WX_plus_b = array(HL_WX_plus_b).reshape(-1,1)
	HL_WX_plus_b = minmax_scale.fit_transform(HL_WX_plus_b)
	HL_WX_plus_b = numpy.reshape(HL_WX_plus_b,HL_WX_plus_b_shape)

	# applying relu
	HL_values = numpy.clip(HL_WX_plus_b,0.,float("inf"))

	# tt = time.clock() - tt
	# print "FC->H: " + str(tt)

	# ------------------------------------ Hidden Layer --> OUTPUT -------------------------------------

	output_wx_plus_b = numpy.dot(weights_HL_to_output, HL_values) + bhl

	# applying relu 
	output = numpy.clip(output_wx_plus_b,0.,float("inf"))
	print output

	# if iterat_epoch%100 == 0:
	# 	print '---------------------------------------------------------------------------------------'
	# 	print 'Output for image '+str(iterat_image)+' is:'
	# 	print output
	# 	print '---------------------------------------------------------------------------------------'

	# tt = time.clock() - tt
	# print "H->O: " + str(tt)

	# ----------------------------------- END OF FORWARD PROPAGATION -----------------------------------
