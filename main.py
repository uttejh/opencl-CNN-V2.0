import numpy
# import pyopencl as cl 
# from time import time
from PIL import Image
import Image
from procedures import *
import time

numOfFiltersLayer1 = 4
numOfInputs1 = 28*28		#28*28
numOfOutputs1 = 26*26	#26*26
b1 = 1.

# Reads image and converts it to an array - our input
def readImage(x):
	# Open Image
	image = Image.open('1.jpg')

	# Resize each image to one desired shape
	image = image.resize((28,28),Image.ANTIALIAS)

	# # Save as ...., "optimize" - reduce size (bytes)
	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

	# # Convert image into array of pixels
	# arr = array(image)
	# return arr

	data = numpy.asarray( image, dtype="float32" )
	return data

# Initialise filters in convolution layer 1



p = Procedures()

# Creating filters for conv layer1
p.initFilters1(numOfFiltersLayer1, numOfInputs1, numOfOutputs1)

for iterat in range(1):
	# start timer
	start = time.clock()

	# read input
	input_data = readImage(iterat)

	# read labels
	# code this later
	# label = read_from....(label_list.txt)
	# print(input_data)
	# Convolute with input and mention no. of filters to be used
	conv_layer1 = p.convolution(input_data, filters1, b1)
	
	relu1 = p.relu(conv_layer1);
	print(relu1)
	
	# p.test(filters1)

	tt = time.clock() - start
	print(tt)
	# print(input_data)