import numpy
from PIL import Image
import Image
from procedures import *
import time
from numpy import array

numpy.set_printoptions(threshold=numpy.nan)
numOfFiltersLayer1 = 20
numOfFiltersLayer2 = 40

numOfInputs1 = numOfFiltersLayer1*28*28		
numOfOutputs1 = numOfFiltersLayer1*28*28	

numOfInputs2 = numOfFiltersLayer1*numOfFiltersLayer2*14*14		
numOfOutputs2 = numOfFiltersLayer1*numOfFiltersLayer2*14*14		

b1 = 1.
b2 = 1.
bFC = 1.
bH = 1.

alpha = 0.1

# Reads image and converts it to an array - our input
def readImage(x):
	# Open Image
	image = Image.open('1.jpg')

	# Resize each image to one desired shape
	image = image.resize((28,28),Image.ANTIALIAS)

	# # Save as ...., "optimize" - reduce size (bytes)
	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

	data = numpy.array( image, dtype="double" ) 
	return data


p = Procedures()

filters1 = []
filters2 = []
fsize = 3
# Creating filters for conv layer1
filters1 = p.initFilters1(numOfFiltersLayer1, numOfInputs1, numOfOutputs1, fsize)

filters2 = p.initFilters2(numOfFiltersLayer2, numOfInputs2, numOfOutputs2, fsize)

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
	if iterat == 0:
		numOfHiddenNeurons = 100
		numOfOutputNeurons = 10

		n_in1 = FC.shape[0]
		n_out1 = numOfHiddenNeurons
		w_bound1 = numpy.sqrt(6./float(n_in1+n_out1))
		weights_FC_to_HL = numpy.random.uniform(-w_bound1,w_bound1,(numOfHiddenNeurons, n_in1))

		n_in2 = numOfHiddenNeurons
		n_out2 = numOfOutputNeurons
		w_bound2 = numpy.sqrt(6./float(n_in2+n_out2))
		weights_HL_to_output = numpy.random.uniform(-w_bound2,w_bound2,(numOfOutputNeurons, numOfHiddenNeurons))
	
	HL_WX_plus_b = numpy.dot(weights_FC_to_HL, FC) + bFC

	# applying relu
	HL_values = numpy.clip(HL_WX_plus_b,0.,float("inf"))

	# tt = time.clock() - tt
	# print "FC->H: " + str(tt)

	# ------------------------------------ Hidden Layer --> OUTPUT -------------------------------------

	output_wx_plus_b = numpy.dot(weights_HL_to_output, HL_values) + bH

	# applying relu 
	output = numpy.clip(output_wx_plus_b,0.,float("inf"))

	# tt = time.clock() - tt
	# print "H->O: " + str(tt)

	# ----------------------------------- END OF FORWARD PROPAGATION -----------------------------------


	# --------------------------------------------------------------------------------------------------
	# ---------------------------------------- BACK PROPAGATION ----------------------------------------
	# --------------------------------------------------------------------------------------------------

	# --------------------------------------------------------------------------------------------------
	# -------- [ CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 <-- FC <-- HIDDEN LAYER <-- OUTPUT ] ------
	# --------------------------------------------------------------------------------------------------



	# --------------------------------------------- ERROR ----------------------------------------------

	# error = []
	# label = 3
	# for ii in range(numOfOutputNeurons):
	# 	if ii == label:
	# 		target = 1.0
	# 	else:
	# 		target = 0.0 
	# 	error.append(0.5*(target - output[ii])**2)


	# ------------------------------------- HIDDEN LAYER <-- OUTPUT ------------------------------------

	# Calculating errors for each weight (10*100 weights) at HL
	# Dw_HL_to_output = 0
	temp_err = []
	err = 0
	for i in range(numOfOutputNeurons):
		
		# ---------------------------------------- ERROR -----------------------------------------------

		# Dummy label
		label = 3
		if i == label:
			target = 1.0
		else:
			target = 0.0
		err = 0.5*(target - output[i])**2

		derivative = 1.0 if output[i] > 0 else 0.0

		temp = []
		for j in range(numOfHiddenNeurons):

			# E*f`(x)*w 
			Dw_HL_to_output = (err*derivative*weights_HL_to_output[i][j])

			# ommitted minus from E*f`(x)*w so that 
			# [weights_HL_to_output = weights_HL_to_output - (-DW)] becomes
			# [weights_HL_to_output = weights_HL_to_output + DW]
			weights_HL_to_output[i][j] += alpha*Dw_HL_to_output

			# appending the omitted -ve sign
			temp.append(-1*Dw_HL_to_output)

		temp_err.append(temp)

	# -------------------------------- FC <-- HIDDEN LAYER ------------------------------------

	# error at hidden layer
	global_error = []

	for i in range(numOfHiddenNeurons):
		tempge = 0
		for j in range(numOfOutputNeurons):
			tempge += temp_err[j][i]
		global_error.append(tempge)


	# updated weights
	weights_FC_to_HL, temp_FC_err = p.BP_FC_to_HL(numOfHiddenNeurons, n_in1, global_error, HL_values, weights_FC_to_HL, alpha)

	temp_FC_err = numpy.transpose(temp_FC_err)

	
	# ---------------------------- CONVOLUTION LAYER 2 <-- FC ---------------------------------
	
	# error at FC
	# global_error_FC=[]
	loc_err = 0
	err_into_derivative = []

	for i in range(n_in1):
		loc_err = numpy.sum(temp_FC_err[i])
		# global_error_FC.append(loc_err)
		err_into_derivative.append(loc_err*(1.0 if FC[i] > 0 else 0.0))

	pool2_len = pool2_shape[1]
	size_pool2 = pool2_len**2
	# tempnum = numOfFiltersLayer2*size_pool2
	range2 = numOfFiltersLayer1*size_pool2
	
	for i in range(numOfFiltersLayer2):
		row = i*range2
		
		temp = 0
		for j in range(range2):
			temp += err_into_derivative[row + j]

		filto2 = filters2[i]
		# filters2 weight update
		filters2[i] = filto2 - alpha*temp*filto2


	# ---------------------- CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 --------------------

	# reshape errIntoDerivative
	# 3920 = 40*20*7*7
	err_into_der_reshape = numpy.reshape(err_into_derivative, (numOfFiltersLayer2,numOfFiltersLayer1,pool2_len*pool2_len))

	# reshape index2. Same as above 
	index2_reshape = numpy.reshape(index2,(numOfFiltersLayer2,numOfFiltersLayer1,pool2_len*pool2_len))

	# print index2_reshape[0][0]
	# find global error
	# global_error_conv2 = p.conv_global_error(err_into_der_reshape, index2_reshape, filters2, numOfFiltersLayer2, numOfFiltersLayer1, (relu2_shape[2],relu2_shape[2]), fsize)

	# print global_error_conv2[0][1][0]
	# print global_error_conv2[1][1][0]
	# print global_error_conv2[39][0][0]
	# for i in range(20):
	# 	print global_error_conv2[i][0][0]
		# print index2_reshape[i][1][0]\


	# d=[]
	# for ii in range(numOfFiltersLayer2):
	# 	for jj in range(numOfFiltersLayer1):
	# 		for i in range(relu2_shape[2]+fsize-1):
	# 			for j in range(relu2_shape[2]+fsize-1): 
	# 				for k in range(fsize):
	# 					for l in range(fsize):
	# 						d.append(err_into_der_reshape[ii][jj])

	# new_err_der = p.fill_zeros(index2_reshape, err_into_der_reshape, numOfFiltersLayer2, numOfFiltersLayer1, relu2_shape[2])
	errdernew = []
	for i in range(numOfFiltersLayer2):
		tem1 = []
		for j in range(numOfFiltersLayer1):
			tomodify = numpy.zeros((pool2_len*pool2_len*4)).astype(numpy.float64)
			index2_reshape[i][j].sort(axis=0)
			# index2_reshape[i][j].sort(axis=1)

			xx=index2_reshape[i][j].astype(int)
			# yy=err_into_der_reshape[i][j].astype(numpy.float64)
			yy=err_into_der_reshape[i][j]
			for (ind, rep) in zip(xx, yy):
				tomodify[ind] = rep
			tem1.append(tomodify)
			# print tomodify.astype(numpy.float64)
		errdernew.append(tem1)
	# print array(errdernew).dtype
	# print array(index2_reshape).shape
	errdernew = numpy.reshape(errdernew, (numOfFiltersLayer2,numOfFiltersLayer1,14,14))


	global_error_conv2 = p.conv_global_error(errdernew, filters2, numOfFiltersLayer2, numOfFiltersLayer1, (relu2_shape[2],relu2_shape[2]), fsize)

	global_error_conv2 = numpy.transpose(global_error_conv2,(1,2,3,0))

	err_conv1_pad2 =  global_error_conv2.sum(axis=3)
	err_conv1_pad2_shape = array(err_conv1_pad2).shape
	print err_conv1_pad2[0][0]
	conv1_depad = p.depad(err_conv1_pad2, numOfFiltersLayer1, err_conv1_pad2_shape[1] ,fsize)
	print conv1_depad[0][0]
	print err_conv1_pad2_shape
	tt = time.clock() - start
	print(tt)



