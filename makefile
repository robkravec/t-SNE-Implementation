writeup.pdf: write.ipynb, plots
    jupyter nbconvert ... which makes .ipynb file into a pdf

plots: datasets, make_plots.py (probably 2 separate statements)
    make plots directory if it doesn't exist
    run make_plots.py which uses the datasets

datasets: data/get_data.py
   run the get_data script, which creates datasets