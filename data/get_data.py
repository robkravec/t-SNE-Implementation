import data.data_utilities as util
from sklearn.datasets import fetch_olivetti_faces

###################################################################################
# Downloading real world data sets
###################################################################################


# Download UCI PANCAN data set
util.get_uci()  
# Download Allen Institute Human M1 10x data set
util.get_humanM1_10x()
# Download MNIST data set
util.get_mnist()
# Download COIL-20 data set
util.get_coil20()
# Download Olivetti faces data set through sklearn api
fetch_olivetti_faces()

# Process COIL-20 data set into relevant features
util.proc_coil20()