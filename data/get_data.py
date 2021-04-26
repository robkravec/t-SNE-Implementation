import data_utilities as util
import os, datetime 
from sklearn.datasets import fetch_olivetti_faces

###################################################################################
# Downloading real world data sets
###################################################################################


# Download UCI PANCAN data set
util.get_uci()  
# Download Allen Institute Human M1 10x data set
util.get_humanM1_10x()
# Download MNIST data set
# util.get_mnist() # link to data is glitchy 
# Download COIL-20 data set
util.get_coil20()
# Download Olivetti faces data set through sklearn api
fetch_olivetti_faces()

# Process COIL-20 data set into relevant features
util.proc_coil20()

# Set timestamp for TCGA data to current time 
now = datetime.datetime.now().timestamp()
os.utime('TCGA-PANCAN-HiSeq-801x20531/data.csv', (now, now))
os.utime('TCGA-PANCAN-HiSeq-801x20531/labels.csv', (now, now))
