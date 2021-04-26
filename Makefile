writeup.pdf: writeup.ipynb data/plots/plot1.png data/plots/plot2.png
	jupyter nbconvert --to pdf --no-input writeup.ipynb
   
data/plots/plot2.png: data/humanM1_10x/matrix.csv data/humanM1_10x/metadata.csv data/TCGA-PANCAN-HiSeq-801x20531/data.csv data/TCGA-PANCAN-HiSeq-801x20531/labels.csv data/makeplot2.py
	mkdir -p data/plots
	cd data; python3 makeplot2.py

data/plots/plot1.png: data/coil-20-proc/X_COIL.npy data/coil-20-proc/y_COIL.csv data/mnist/X_MNIST.npy data/mnist/y_MNIST.npy data/makeplot1.py
	mkdir -p data/plots
	cd data; python3 makeplot1.py

data/coil-20-proc/X_COIL.npy data/coil-20-proc/y_COIL.csv data/mnist/X_MNIST.npy data/mnist/y_MNIST.npy data/humanM1_10x/matrix.csv data/humanM1_10x/metadata.csv data/TCGA-PANCAN-HiSeq-801x20531/data.csv data/TCGA-PANCAN-HiSeq-801x20531/labels.csv: data/get_data.py data/data_utilities.py
	cd data; python3 get_data.py

clean_pdf:
	rm writeup.pdf

clean_plots:
	rm -r data/plots/
    
clean_data:
	rm -r data/coil-20-proc
	rm -r data/mnist
	rm -r data/humanM1_10x
	rm -r data/TCGA-PANCAN-HiSeq-801x20531

.PHONY: clean plots; clean_pdf; clean_data;