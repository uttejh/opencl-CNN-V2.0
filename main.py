import numpy
from PIL import Image
import Image
from procedures import *
import time
from numpy import array
from sklearn.preprocessing import MinMaxScaler

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

for iterat in range(20):
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
	
	HL_WX_plus_b = numpy.dot(weights_FC_to_HL, FC) #+ bFC

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

	output_wx_plus_b = numpy.dot(weights_HL_to_output, HL_values) #+ bH

	# applying relu 
	output = numpy.clip(output_wx_plus_b,0.,float("inf"))

	print output

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

	# normalize
	# minmax_scale = MinMaxScaler(feature_range=(-1, 1), copy=True)

	error = []
	loss = []
	label = 3
	for i in range(numOfOutputNeurons):
		label = 3
		if i == label:
			target = 1.0
		else:
			target = 0.0
		e = (target - output[i])
		loss.append(0.5*e**2)
		error.append(e)

	totalloss = numpy.sum(loss)

	# normalize
	error_shape = array(error).shape
	error = array(error).reshape(-1,1)
	error = minmax_scale.fit_transform(error)
	error = numpy.reshape(error,error_shape)
	
	# ------------------------------------- HIDDEN LAYER <-- OUTPUT ------------------------------------

	# Calculating errors for each weight (10*100 weights) at HL
	# Dw_HL_to_output = 0
	temp_err = []
	err = 0
	for i in range(numOfOutputNeurons):
		
		# ---------------------------------------- ERROR -----------------------------------------------

		# Dummy label
		# label = 3
		# if i == label:
		# 	target = 1.0
		# else:
		# 	target = 0.0
		# # err = 0.5*(target - output[i])**2
		# err = (target - output[i])
		derivative = 1.0 if output[i] > 0 else 0.0
		# print err
		temp = []
		for j in range(numOfHiddenNeurons):

			# E*f`(x)*w 
			Dw_HL_to_output = (error[i]*derivative*weights_HL_to_output[i][j])
			# ommitted minus from E*f`(x)*w so that 
			# [weights_HL_to_output = weights_HL_to_output - (-DW)] becomes
			# [weights_HL_to_output = weights_HL_to_output + DW]
			# change1
			weights_HL_to_output[i][j] += alpha*error[i]*derivative*HL_values[j]

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


	# normalizing
	global_error_shape = array(global_error).shape
	global_error = array(global_error).reshape(-1,1)
	global_error = minmax_scale.fit_transform(global_error)
	global_error = numpy.reshape(global_error,global_error_shape)


	# updated weights
	weights_FC_to_HL, temp_FC_err = p.BP_FC_to_HL(numOfHiddenNeurons, n_in1, global_error, HL_values, weights_FC_to_HL, alpha, FC)

	temp_FC_err = numpy.transpose(temp_FC_err)

	# ---------------------------- CONVOLUTION LAYER 2 <-- FC ---------------------------------
	
	glob_err = []
	for i in range(n_in1):
		glob_err.append(numpy.sum(temp_FC_err[i]))

	# normalizing
	global_err_shape = array(glob_err).shape
	glob_err = array(glob_err).reshape(-1,1)
	glob_err = minmax_scale.fit_transform(glob_err)
	glob_err = numpy.reshape(glob_err,global_err_shape)
	

	pool2_len = pool2_shape[1]
	size_pool2 = pool2_len**2


	index2_reshape = numpy.reshape(index2,(numOfFiltersLayer2,numOfFiltersLayer1,pool2_len*pool2_len))

	glob_err_reshape = numpy.reshape(glob_err, (numOfFiltersLayer2,numOfFiltersLayer1,pool2_len*pool2_len))

	FCreshape = numpy.reshape(FC, (numOfFiltersLayer2,numOfFiltersLayer1,pool2_len,pool2_len))

	globerrnew = []
	fdash_c2=[]
	for i in range(numOfFiltersLayer2):
		tem1 = []
		t2fdash = []
		for j in range(numOfFiltersLayer1):
			temp_fdash_c2 = []
			for k in range(pool2_len):
				for l in range(pool2_len):
					temp_fdash_c2.append(1.0 if FCreshape[i][j][k][l] > 0 else 0.0)

			# t2fdash.append(temp_fdash_c2)
			zz = temp_fdash_c2
			tomodify1 = numpy.zeros((pool2_len*pool2_len*4)).astype(numpy.float64)
			tomodify2 = numpy.zeros((pool2_len*pool2_len*4)).astype(numpy.float64)

			index2_reshape[i][j].sort(axis=0)
			# index2_reshape[i][j].sort(axis=1)

			xx=index2_reshape[i][j].astype(int)
			# yy=err_into_der_reshape[i][j].astype(numpy.float64)
			yy=glob_err_reshape[i][j]
			for (ind, rep, fd) in zip(xx, yy, zz):
				tomodify1[ind] = rep
				tomodify2[ind] = fd
			tem1.append(tomodify1)
			t2fdash.append(tomodify2)
		
		# 40*20*196
		globerrnew.append(tem1)
		fdash_c2.append(t2fdash)

	conv2_reshape = numpy.reshape(conv_layer2,(numOfFiltersLayer2,numOfFiltersLayer1,relu2_shape[2]*relu2_shape[2]))

	globerrintoder = array(globerrnew)*array(fdash_c2)
	globerrintoder = numpy.reshape(globerrintoder,(numOfFiltersLayer2,numOfFiltersLayer1,pool2_len*2,pool2_len*2))
	# totalgloberrintoder = numpy.sum(globerrintoder_reshape,axis=1)

	globerrintoderintox = []
	for i in range(numOfFiltersLayer2):
		temporrr = globerrintoder[i]*array(pool1)
		globerrintoderintox.append(temporrr)

	# globerrintoderintox = globerrintoder*array(conv2_reshape)
	globerrintoderintox = numpy.reshape(globerrintoderintox,(numOfFiltersLayer2,numOfFiltersLayer1*pool2_len*pool2_len*4))
	totalerr = numpy.sum(globerrintoderintox,axis=1)

	# normalizing
	totalerr_shape = array(totalerr).shape
	totalerr = array(totalerr).reshape(-1,1)
	totalerr = minmax_scale.fit_transform(totalerr)
	totalerr = numpy.reshape(totalerr,totalerr_shape)

	globerrintoderintow_c2=[]
	for i in range(numOfFiltersLayer2):
		filto2 = filters2[i]
		# filters2 weight update
		filters2[i] = filto2 - alpha*(totalerr[i]) 

		globerrintoderintow_c2.append(globerrintoder[i]*(numpy.sum(filto2)))

	# # normalizing
	# globerrintoderintow_c2_shape = array(globerrintoderintow_c2).shape
	# globerrintoderintow_c2 = array(globerrintoderintow_c2).reshape(-1,1)
	# globerrintoderintow_c2 = minmax_scale.fit_transform(globerrintoderintow_c2)
	# globerrintoderintow_c2 = numpy.reshape(globerrintoderintow_c2,globerrintoderintow_c2_shape)

	# print globerrintoderintow_c2[0]
	# ----------------- CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2  --------------------------


	globerrintoderreshape = numpy.reshape(globerrintoderintow_c2, (numOfFiltersLayer2,numOfFiltersLayer1,relu2_shape[2],relu2_shape[2]))

	global_error_conv2 = p.conv_global_error(globerrintoderreshape, filters2, numOfFiltersLayer2, numOfFiltersLayer1, (relu2_shape[2],relu2_shape[2]), fsize)

	# global_error_conv2 = numpy.clip(global_error_conv2,-100.,100.)

	global_error_conv2 = numpy.transpose(global_error_conv2,(1,2,3,0))
	global_temp_sum = numpy.sum(global_error_conv2,axis=3)
	# global_temp_sum = numpy.clip(global_temp_sum,-100.,100.)
	# print global_temp_sum
	# normalizing
	global_temp_sum_shape = array(global_temp_sum).shape
	global_temp_sum = array(global_temp_sum).reshape(-1,1)
	global_temp_sum = minmax_scale.fit_transform(global_temp_sum)
	global_temp_sum = numpy.reshape(global_temp_sum,global_temp_sum_shape)

	ge_slice = global_temp_sum[:,1:-1,1:-1]

	ge_slicereshape = numpy.reshape(ge_slice,(numOfFiltersLayer1,relu2_shape[2]*relu2_shape[2]))

	index1_reshape = numpy.reshape(index1,(numOfFiltersLayer1,relu2_shape[2]*relu2_shape[2]))

	fdash_c1=[]
	ge_c1 = []
	for i in range(numOfFiltersLayer1):
		temp_fdash_c1 = []
		for j in range(conv1_shape[2]):
			for k in range(conv1_shape[2]):
				temp_fdash_c1.append(1.0 if relu1[i][j][k] > 0 else 0.0)
		fdash_c1.append(temp_fdash_c1)

		tomodify1 = numpy.zeros((conv1_shape[2]*conv1_shape[2])).astype(numpy.float64)
		index1_reshape[i].sort(axis=0)
			# index2_reshape[i][j].sort(axis=1)

		xx=index1_reshape[i].astype(int)
			# yy=err_into_der_reshape[i][j].astype(numpy.float64)
		yy=ge_slicereshape[i]
		for (ind, rep) in zip(xx, yy):
			tomodify1[ind] = rep
		ge_c1.append(tomodify1)

	globerrintoder_c1 = array(ge_c1)*array(fdash_c1)

	input_data_reshape = numpy.reshape(input_data, (input_shape1[0]*input_shape1[0]))

	globerrintoderx_c1 = []
	for i in range(numOfFiltersLayer1):
		tempp = globerrintoder_c1[i]*input_data_reshape
		globerrintoderx_c1.append(tempp)

	totalerr_c1 = numpy.sum(globerrintoderx_c1,axis=1)

	# normalizing
	totalerr_c1_shape = array(totalerr_c1).shape
	totalerr_c1 = array(totalerr_c1).reshape(-1,1)
	totalerr_c1 = minmax_scale.fit_transform(totalerr_c1)
	totalerr_c1 = numpy.reshape(totalerr_c1,totalerr_c1_shape)

	for i in range(numOfFiltersLayer1):
		filto1 = filters1[i]
		# filters2 weight update
		filters1[i] = filto1 - alpha*(totalerr_c1[i])


	

	tt = time.clock() - start
	print(tt)



