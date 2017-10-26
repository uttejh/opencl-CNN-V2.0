import numpy
import time


a = numpy.ones((28,28)).astype(numpy.float32)

#x = a.get()
b = numpy.ones((28,28)).astype(numpy.float32)
#y = b.get()
#dot_ab_gpu = cl_array.dot(a,b).get()
start = time.clock()
for x in range(512):
	aaa = numpy.dot(a,b)

end = time.clock() - start
print end