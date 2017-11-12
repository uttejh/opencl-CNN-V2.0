import numpy
from PIL import Image
import Image
from procedures import *
import time
from numpy import array

numpy.set_printoptions(threshold=numpy.nan)
numOfFiltersLayer1 = 20
numOfFiltersLayer2 = 40

numOfInputs1 = numOfFiltersLayer1 * 28 * 28
numOfOutputs1 = numOfFiltersLayer1 * 28 * 28

numOfInputs2 = numOfFiltersLayer1 * numOfFiltersLayer2 * 14 * 14
numOfOutputs2 = numOfFiltersLayer1 * numOfFiltersLayer2 * 14 * 14

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
	image = image.resize((28, 28), Image.ANTIALIAS)

	# # Save as ...., "optimize" - reduce size (bytes)
	# image.save("image_scaled_opt.jpg",optimize=True,quality=95)

	data = numpy.array(image, dtype="double")
	return data


p = Procedures()

filters1 = []
filters2 = []
fsize = 3

p.initiateWeightsToFile(numOfFiltersLayer1, numOfInputs1, numOfOutputs1, fsize)

# Creating filters for conv layer1
filters1 = p.GetsFilters1()
'''
filters = open("filter1.txt", "r")  # opens file with name of "filter1.txt"
for line in filters:
    filters1.append(line)


filters = open("filter2.txt", "r")  # opens file "filter2.txt"
for line in filters:
    filters2.append(line)


'''
filters2 = p.initFilters2(
	numOfFiltersLayer2, numOfInputs2, numOfOutputs2, fsize)


