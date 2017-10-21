import numpy
import time
from procedures import *
# from PIL import Image
# from numpy import array

# image = Image.open('3.jpg')
# # Converts image into array of pixels

# image = image.resize((28,28),Image.ANTIALIAS)
# image.save("image_scaled_opt.jpg",optimize=True,quality=95)
# arr = array(image)
# print arr.shape
# start = time.clock()
# filters = dict()
# for x in range(256):
# 	filters[x] = []
# 	filters[x] = numpy.random.uniform(-0.5,0.5,(3,3,3))
	
# # print(filters[1])
# # print(filters[4])
# tt = time.clock() - start
# print('Time:'+str(tt))

p = Procedures()

p.initFilters1(128,9,1)
print(filters1[1])