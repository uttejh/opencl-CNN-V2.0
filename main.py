import numpy
# import pyopencl as cl 
from time import time
from PIL import Image
import Image
from procedures import *


numOfFiltersLayer1 = 4
numOfInputs1 = 2352		#28*28*3
numOfOutputs2 = 2028	#26*26*3

# Reads image and converts it to an array - our input
def readImage(x):
	# Open Image
	image = Image.open('3.jpg')

	# Resize each image to one desired shape
	image = image.resize((28,28),Image.ANTIALIAS)

	# # Save as ...., "optimize" - reduce size (bytes)
	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

	# # Convert image into array of pixels
	# arr = array(image)
	# return arr

	data = numpy.asarray( image, dtype="int32" )
	return data

# Initialise filters in convolution layer 1



p = Procedures()

# Creating filters for conv layer1
p.initFilters1(numOfFiltersLayer1, numOfInputs1, numOfOutputs2)

for iterat in range(1):
	# start timer
	# start = time.clock()

	# read input
	input_data = readImage(iterat)

	# read labels
	# code this later
	# label = read_from....(label_list.txt)

	# Convolute with input and mention no. of filters to be used
	conv_layer1 = p.convolution(input_data)

	print(conv_layer1)
	# print(input_data)