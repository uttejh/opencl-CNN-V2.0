import numpy
import pickle
from numpy import array
# import matplotlib.pyplot as plt
# x=[]
# for i in range(20):
# 	x.append(i)
# y = numpy.random.uniform(0,18,(20))

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# plt.xticks(numpy.arange(x.min(), 2000, 0.1))
# ax1.clear()    #Remove this if you want to see distict colour graph at each interval.
# ax1.set_xlim([0, 200]) #max and min value to X. Updates at each new instance.
# ax1.set_ylim([0, 20]) # max and min value value to Y. Constant throughout.
# ax1.plot(x,y)
# plt.show

import matplotlib.pyplot as plt

# # plt.plot(x, y)
# # plt.axis([0, 6, 0, 10])
# # plt.show()
f = open('./weights/totalloss.txt')
y = pickle.load(f)
f.close()
x=numpy.arange(0,2000)
# # x = [0,5,9,10,15]
# # y = [0,1,2,3,4]
# plt.plot(x,y)
# plt.xticks(numpy.arange(min(x), max(x)+2, 10.0))
# plt.show()
# x=numpy.arange(0,19,1.0)
# print x
re_y = []
for i in range(2000):
	# for j in range(10):
	re_y.append(numpy.sum(y[i:i+10]))



y=array(re_y).ravel()
# y=re_y[::-1]
plt.plot(x,y)
plt.xticks(numpy.arange(min(x), max(x)+2, 200.0))
plt.title('Loss VS Epochs',fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss',fontsize=14)

plt.show()
# print y