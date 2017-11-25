import numpy
# import matplotlib.pyplot as plt
x=[]
for i in range(20):
	x.append(i)
y = numpy.random.uniform(0,18,(20))

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# plt.xticks(numpy.arange(x.min(), 2000, 0.1))
# ax1.clear()    #Remove this if you want to see distict colour graph at each interval.
# ax1.set_xlim([0, 200]) #max and min value to X. Updates at each new instance.
# ax1.set_ylim([0, 20]) # max and min value value to Y. Constant throughout.
# ax1.plot(x,y)
# plt.show

# import matplotlib.pyplot as plt

# # plt.plot(x, y)
# # plt.axis([0, 6, 0, 10])
# # plt.show()


# # x = [0,5,9,10,15]
# # y = [0,1,2,3,4]
# plt.plot(x,y)
# plt.xticks(numpy.arange(min(x), max(x)+2, 10.0))
# plt.show()
x=numpy.arange(0,19,1.0)
print x