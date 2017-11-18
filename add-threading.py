import numpy
from threading import Thread
from multiprocessing.pool import ThreadPool
import time
from numpy import array

arr = numpy.random.rand(39200,100)

def add(x):
	return numpy.sum(x)
start = time.clock()
pool = ThreadPool(processes=1)
result = pool.map(add, (f for f in arr))
pool.close()
print time.clock() - start
print array(result).shape 