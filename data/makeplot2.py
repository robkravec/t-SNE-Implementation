from data_utilities import *
from sklearn.datasets import fetch_olivetti_faces
import os 



###########################################################################################
# Generates plots comparing t-SNE, PCA, and Isomap on the PANCAN and Human M1 10x data sets 
###########################################################################################


if not os.path.exists("./plots"):
    os.mkdir("./plots")
    
save_path = "./plots/plot2.png"


Xcan, ycan = load_UCI('TCGA-PANCAN-HiSeq-801x20531')
Xm1, ym1   = load_humanM1('humanM1_10x', nrows=4000)
ym1 = ym1[:,1].copy()


colors = ['red', 'mediumspringgreen', 'royalblue','orange' ,'deeppink',
          'aqua','blue', 'brown' ,'fuchsia', 'lime']
shapes = ['o', '+', 'x', '*', 'D', 'p']

fmt_list = [gen_fmt_dict(y, colors, shapes) for y in [ycan, ym1]]

titles = ["PANCAN", "Human M1 10x"]
legends = [True, False]

compare_methods_2d(data_list=[Xcan, Xm1], label_list=[ycan, ym1], data_names=titles, 
                   fmt_list=fmt_list, legends=legends, niter=1000)

plt.savefig(save_path, format = 'png')