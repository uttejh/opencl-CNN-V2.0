import numpy
# import pyopencl as cl 
# from time import time
from PIL import Image
import Image
from procedures import *
import time
from numpy import array

numOfFiltersLayer1 = 4
numOfFiltersLayer2 = 8

numOfInputs1 = numOfFiltersLayer1*numOfFiltersLayer1*28*28		
numOfOutputs1 = numOfFiltersLayer1*numOfFiltersLayer1*28*28	

numOfInputs2 = numOfFiltersLayer1*numOfFiltersLayer1*14*14		
numOfOutputs2 = numOfFiltersLayer1*numOfFiltersLayer1*14*14		

b1 = 1.
b2 = 1.

# Reads image and converts it to an array - our input
def readImage(x):
	# Open Image
	image = Image.open('1.jpg')

	# Resize each image to one desired shape
	# image = image.resize((28,28),Image.ANTIALIAS)

	# # Save as ...., "optimize" - reduce size (bytes)
	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

	data = numpy.asarray( image, dtype="float32" )
	return data


p = Procedures()

# Creating filters for conv layer1
p.initFilters1(numOfFiltersLayer1, numOfInputs1, numOfOutputs1)

p.initFilters2(numOfFiltersLayer2, numOfInputs2, numOfOutputs2)

for iterat in range(1):
	start = time.clock()

	# -------------------------------------- READ INPUT -------------------------------------------

	input_data = readImage(iterat)
	
	# read labels
	# code this later
	# label = read_from....(label_list.txt)
	
	# Get shape and size/order
	input_shape1 = input_data.shape
	numinputs_pad1 = 1
	order_pad1 = input_shape1[0]

	# -----------------------------------------------------------------------------------------------
	#                     [ PADDING --> CONVOLUTION ] --> RELU --> POOLING (FIRST ITERATION)
	# -----------------------------------------------------------------------------------------------


	# -----------------------------------------PADDING-----------------------------------------------

	pad1 = p.pad(input_data, numinputs_pad1, order_pad1)
	pad_arr1 = array(pad1)
	pad_shape1 = pad_arr1.shape

	# ---------------------------------------CONVOLUTION---------------------------------------------

	numinputs_conv1 = 1
	order_conv1 = pad_shape1[1]

	# Convolute with input and mention no. of filters to be used
	conv_layer1 = p.convolution(input_data, filters1, b1, numinputs_conv1, order_conv1)	
	conv1_shape = array(conv_layer1).shape

	print conv_layer1[0]

	# ------------------------------------------RELU--------------------------------------------------

	numinputs_relu1 = conv1_shape[0]
	order_relu1 = conv1_shape[1]

	relu1 = p.relu(conv_layer1, numinputs_relu1, order_relu1)

	relu_arr1 = array(relu1)
	relu1_shape = relu_arr1.shape

	# -----------------------------------------POOLING-------------------------------------------------

	numinputs_pool1 = relu1_shape[0]
	order_pool1 = relu1_shape[1]
	
	pool1 = p.pooling(relu1, numinputs_pool1, order_pool1)

	pool_arr1 = array(pool1)
	pool1_shape = pool_arr1.shape


	# -----------------------------------------------------------------------------------------------
	#                     [ PADDING --> CONVOLUTION ] --> RELU --> POOLING (SECOND ITERATION)
	# -----------------------------------------------------------------------------------------------


	# -----------------------------------------PADDING-----------------------------------------------

	numinputs_pad2 = pool1_shape[0]
	order_pad2 = pool1_shape[1]

	pad2 = p.pad(pool1, numinputs_pad2, order_pad2)

	pad2_shape = array(pad2).shape

	# ---------------------------------------CONVOLUTION---------------------------------------------

	numinputs_conv2 = pad2_shape[0]
	order_conv2 = pad2_shape[1]

	# Convolute with input and mention no. of filters to be used
	conv_layer2 = p.convolution(pad2, filters2, b2, numinputs_conv2, order_conv2)	
	conv2_shape = array(conv_layer2).shape

	# ------------------------------------------RELU--------------------------------------------------

	numinputs_relu2 = conv2_shape[0]
	order_relu2 = conv2_shape[1]

	relu2 = p.relu(conv_layer2, numinputs_relu2, order_relu2)

	relu_arr2 = array(relu2)
	relu2_shape = relu_arr2.shape		

	# -----------------------------------------POOLING-------------------------------------------------

	numinputs_pool2 = relu2_shape[0]
	order_pool2 = relu2_shape[1]
	
	pool2 = p.pooling(relu2, numinputs_pool2, order_pool2)

	pool_arr2 = array(pool2)
	pool2_shape = pool_arr2.shape

	# -----------------------------------END OF SECOND ITERATION----------------------------------------

	# ----------------------------------- FULLY CONNECTED LAYER ----------------------------------------

	FC = array(pool2).ravel()

	tt = time.clock() - start
	print(tt)