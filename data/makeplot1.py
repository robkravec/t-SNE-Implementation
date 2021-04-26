from data_utilities import *
from sklearn.datasets import fetch_olivetti_faces
import os 



###################################################################################
# Generates plots comparing t-SNE, PCA, and Isomap on the MNIST, COIL-20, 
# and Olivetti faces data sets 
###################################################################################


if not os.path.exists("./plots"):
    os.mkdir("plots")
    
save_path = "plots/plot1.png"


Xm, ym = load_mnist('mnist', size=6000)
Xc, Yc = load_coil20('coil-20-proc')
yc = Yc[:,0]
olivetti = fetch_olivetti_faces()
Xo = olivetti.data
yo = olivetti.target


colors = ['red', 'mediumspringgreen', 'royalblue','orange' ,'deeppink',
          'aqua','blue', 'brown' ,'fuchsia', 'lime']
shapes = ['o', '+', 'x', '*', 'D', 'p']

fmt_list = [gen_fmt_dict(y, colors, shapes) for y in [ym, yc, yo]]

titles = ["MNIST Digits", "COIL-20", "Olivetti Face's"]
legends = [False, False, False]

compare_methods_2d(data_list=[Xm, Xc, Xo], label_list=[ym, yc, yo], data_names=titles, 
                   fmt_list=fmt_list, legends=legends, perplexity=40, n_neighbors=12, niter=1000)

plt.savefig(save_path, format = 'png')
