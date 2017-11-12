import numpy
import pyopencl as cl 
import os
from numpy import array
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

from numpy import array 
# Creating a dictionary
# It consists of our filter weights. Like an array of 3D arrays 
# filters1 = []
# filters2 = []

class Procedures:
	def __init__(self):
		self.bla = []

    def initiateWeightsToFile(filternum, n_in,n_out,fsize):
        #file = open(“testfile.txt”,”w”)  
        w_bound = numpy.sqrt(6./float(n_in+n_out))
		filters1.append(numpy.random.uniform(-w_bound,w_bound,(fsize,fsize)))
        #filters.close()
        np.savetxt("filter1.txt", filters1)


	# Layer 1 filters
	@staticmethod
	def GetFilters1(): #This file is extracted and set everytime we see some changes to filter1 file after backpropagation
		#extract from file
        filters1 = []
		filters = open("filter1.txt", "r")  # opens file with name of "filter1.txt"
        for line in filters:
            for token in line.split():
                filters1.append(token)
        #for x in range(filternum):
		filters.close()	
		#w_bound = numpy.sqrt(6./float(n_in+n_out))
		#filters1.append(numpy.random.uniform(-w_bound,w_bound,(fsize,fsize)))
		
        return filters1

	