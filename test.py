import numpy
import time



x =numpy.random.uniform(-1,1,(10000))
y = numpy.ones((3,3))
start = time.clock()
z = numpy.clip(x,0.,float("inf"))
# print z
print(time.clock() - start)