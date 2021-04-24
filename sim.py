from tsne import tsne
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
from mpl_toolkits import mplot3d
from textwrap import wrap

###################################################################################
# Creating simple shapes.
###################################################################################

def make_two_blobs(n = 500, p = 500, blob1_mean = -2, blob1_sd = 0.5, blob2_mean = 2, blob2_sd = 0.5):
    """Creates two Gaussian blobs with equal numbers of points
    
    Optional inputs: 
    n ~ Total number of data points (half of which will go to each blob)
    p ~ Number of columns (dimensionality) in blob data
    blob1_mean ~ Mean of normal distribution from which blob1 points are drawn
    blob1_sd ~ Standard deviation of normal distribution from which blob1 points are drawn
    blob2_mean ~ Mean of normal distribution from which blob2 points are drawn
    blob2_sd ~ Standard deviation of normal distribution from which blob2 points are drawn
    
    Outputs:
    X ~ n x dim array of blob data
    labs ~ Labels corresponding to the two different blobs"""
    
    X = np.r_[np.random.normal(blob1_mean, blob1_sd, [n // 2, p]), np.random.normal(blob2_mean, blob2_sd, [n // 2, p])]
    labs = np.repeat(np.array([1, 2]), n // 2)
    return X, labs  

def make_line(x = np.linspace(start = 0, stop = 50, num = 200), slope = 2, intercept = 0, noise = 3):
    """Creates single line of points with random scatter
    
    Optional inputs: 
    x ~ Numpy array of x values over which the line should be generated
    slope ~ Slope of line
    intercept ~ Y-intercept of line
    noise ~ Standard deviation of random noise
    
    Outputs:
    X ~ 2 column array with x values in first column and y (line) values in second column"""
    
    return np.c_[x, slope * x + intercept + np.random.normal(loc = 0, scale = 3, size = len(x))]

def make_parallel_lines(x = np.linspace(start = 0, stop = 50, num = 200), num_lines = 2, spacing = 17, \
                        slope = 2, intercept = 0, noise = 3):
    """Creates parallel lines of points with random scatter
    
    Optional inputs: 
    x ~ Numpy array of x values over which the line should be generated
    num_lines ~ Number of parallel lines
    spacing ~ Amount of space between each parallel line
    slope ~ Slope of lines
    intercept ~ Y-intercept of first line plotted
    noise ~ Standard deviation of random noise
    
    Outputs:
    X ~ (2 * len(x)) x (num_lines + 1) array with x values in first column and y (line) values in second column
    labs ~ Labels corresponding to the two different parallel lines"""
    
    # Set original line
    X = make_line(x, slope, intercept, noise)
    
    # Add additional lines until num_lines is reached
    for i in range(1, num_lines):
        X = np.r_[X, make_line(x, slope, intercept + i * spacing, noise)]
    
    # Generate labels
    labs = np.repeat(np.arange(1, num_lines + 1), len(x))
    
    # Return results
    return X, labs

###################################################################################
# Creating more complex, 3-dimensional shapes.
###################################################################################

def make_two_3d_circles(radius = 1, theta = np.linspace(0, 2 * np.pi, 201)):
    """Generate data and labels for 3D linked circles of specified radius. Circles are rotated 90-degrees
    relative to each other and are linked such that one passes through the exact center of the other
    
    Optional inputs: 
    radius ~ Radius of the two circles
    theta ~ Numpy array corresponding to theta values of interest (which, by default, create full circles)
    
    Outputs:
    X ~ (2 * len(theta) x 3) array with coordinates of circles
    labs ~ Labels corresponding to the two different circles"""
    
    # Generate X and labs
    x = np.r_[np.zeros(len(theta)), radius * np.cos(theta)]
    y = np.r_[radius * np.cos(theta), np.zeros(len(theta))]
    z = np.r_[radius * np.sin(theta), radius * np.sin(theta) + radius]
    X = np.c_[x, y, z]
    labs = np.repeat(np.array([1, 2]), len(theta))
    
    # Return results
    return X, labs   

def make_trefoil_knot(size = 1, phi = np.linspace(0, 2 * np.pi, 300)):
    """Generate data and labels for 3D tiefoil knot
    
    Optional inputs: 
    size ~ Scaling factor to increase / decrease size of tiefoil knot
    phi ~ Numpy array corresponding to phi values of interest (which, by default, create full tiefoil knot)
    
    Outputs:
    X ~ (len(theta)) x 3) array with coordinates for tiefoil know
    labs ~ Color labels for tiefoil knot (simply a color gradient, given that there are not distinct classes)"""
    
    # Generate X and labs
    X = size * np.c_[np.sin(phi) + 2 * np.sin(2 * phi), \
             np.cos(phi) - 2 * np.cos(2 * phi), \
             -np.sin(3 * phi)]
    labs = np.repeat(np.array([1, 2, 3]), len(phi) / 3)

    # Return results
    return X, labs   

def make_springs(x_offset = -0.5, y_offset = 0.5, z_offset = 2.5, z = np.linspace(0, 15, 250)):
    """Generate data and labels for two intertwined springs in 3D with specified offsets
    
    Optional inputs:
    x_offset: Offset between springs in x-direction
    y_offset: Offset between springs in y-direction
    z_offset: Offset between springs in z-direction
    z ~ Numpy array corresponding to z values of interest for the "baseline" spring
    
    Outputs:
    X ~ (2 * len(z)) x 3) array with coordinates for tiefoil know
    labs ~ Labels corresponding to the two different springs"""
    
    # Generate X and labs
    labs = np.repeat(np.array([1, 2]), len(z))
    x = np.r_[np.sin(z), np.sin(z) + x_offset]
    y = np.r_[np.cos(z), np.cos(z) + y_offset]
    z = np.r_[z, z + z_offset]
    X = np.c_[x, y, z] 

    # Return results
    return X, labs  

###################################################################################
# Plotting functions.
###################################################################################

def plot_clean_3d(X, labs, title):
    """Generate 3D plotly figure without color bar or axes
    
    Optional inputs: 
    X ~ Coordinates (x, y, z) of figure to be plotted
    labs ~ Labels used to denoted colors in plot
    title ~ Title of plot
    
    Outputs:
    fig ~ The 3d plot"""
    
    # Create 3D plot using plotly
    fig = px.scatter_3d(x = X[:, 0], y = X[:, 1], z = X[:, 2], color = labs)
    fig.update(layout_coloraxis_showscale=False) # Remove color bar
    # Add (and center) title, remove all traces of axes
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        scene=dict(
            xaxis=dict(title = "", showticklabels=False, visible = False),
            yaxis=dict(title = "", showticklabels=False, visible = False),
            zaxis=dict(title = "", showticklabels=False, visible = False)
        )
    )
    
    # Display figure
    return(fig)

def perp_plots(X, labs, perp_vec, ncols = 3, verbose = False, \
               cdict = {1: 'red', 2: 'mediumspringgreen', 3: 'royalblue'}):
    
    """Plots t-SNE output for given dataset and perplexity values
    
    Required inputs: 
    X ~ NxM data matrix
    perp_vec ~ A list of perplexity values
    
    Optional inputs:
    ncols ~ Number of columns in subplot display
    verbose ~ Boolean value to designate whether progress bars for t-SNE should be displayed
    cdict ~ Dictionary of colors to be used for plotting
    
    Outputs:
    Prints grid of plots"""
    
    # Set dimensions of subplots
    nrows = math.ceil(len(perp_vec) / ncols)
    
    # Configure axes
    axes = []
    fig = plt.figure(figsize = (16, 3 * nrows))
    
    # Iteratively generate plots
    for perp in range(len(perp_vec)):
        low_d = tsne(X = X, perplexity = perp_vec[perp], verbose = verbose, optim = "fastest")
        axes.append(fig.add_subplot(nrows, ncols, perp + 1))
        axes[-1].set_title("Perplexity = " + str(perp_vec[perp]))
        plt.scatter(x = low_d[-1, :, 0], y = low_d[-1, :, 1], \
                    edgecolor = None, alpha = 0.8, c = np.array(list(map(lambda x: cdict[x], labs))))
        axes[-1].set_xticklabels([])
        axes[-1].set_yticklabels([])
        axes[-1].xaxis.set_ticks_position('none')
        axes[-1].yaxis.set_ticks_position('none')

def step_plots(X, labs, steps_vec, perplexity = 30, ncols = 3, \
               verbose = False, cdict = {1: 'red', 2: 'mediumspringgreen', 3: 'royalblue'}):
    
    """Plots t-SNE output for given dataset and perplexity values
    
    Required inputs: 
    X ~ NxM data matrix
    steps_vec ~ A list of step numbers
    
    Optional inputs:
    ncols ~ Number of columns in subplot display
    verbose ~ Boolean value to designate whether progress bars for t-SNE should be displayed
    cdict ~ Dictionary of colors to be used for plotting
    
    Outputs:
    Prints grid of plots"""
    
    # Set dimensions of subplots
    nrows = math.ceil(len(steps_vec) / ncols)
    
    # Configure axes
    axes = []
    fig = plt.figure(figsize = (16, 3 * nrows))
    
    # Run t-SNE
    low_d = tsne(X = X, perplexity = perplexity, niter = np.max(steps_vec), verbose = verbose, optim = "fastest")
    
    # Iteratively generate plots
    for step in range(len(steps_vec)):
        axes.append(fig.add_subplot(nrows, ncols, step + 1))
        axes[-1].set_title("Perplexity = " + str(perplexity) + ", Step = " + str(steps_vec[step]))
        plt.scatter(x = low_d[steps_vec[step], :, 0], y = low_d[steps_vec[step], :, 1], \
                    edgecolor = None, alpha = 0.8, c = np.array(list(map(lambda x: cdict[x], labs))))
        axes[-1].set_xticklabels([])
        axes[-1].set_yticklabels([])
        axes[-1].xaxis.set_ticks_position('none')
        axes[-1].yaxis.set_ticks_position('none')

def compare_plots(data_list, title_list, perp_list = [5, 30, 100], step_list = [10, 50, 1000], \
                  perp_plot = True, step_plot_perp = 30, verbose = False, plot_3d = False, \
                  cdict = {1: 'red', 2: 'mediumspringgreen', 3: 'royalblue'}, df = 1):
    
    """Plots t-SNE output for all provided datasets and perplexity values in a grid
    
    Required inputs: 
    data_list ~ List of tuples in the format (X, color labels)
    title_list ~ List of titles for plots in same order as data_list
    
    Optional inputs:
    perp_list ~ List of perplexity values to show for each dataset provided
    step_list ~ List of t-SNE values to show for each dataset provided
    perp_plot ~ Boolean value to denote whether plots should show different plexity values (True) or different
                t-SNE iteration steps (False)
    step_plot_perp ~ Perplexity value to be used if a grid of plots showing t-SNE different iterations is desired
    verbose ~ Boolean value to designate whether progress bars for t-SNE should be displayed
    plot_3d ~ Boolean value to designate whether first row of plots should be in 3D
    cdict ~ Dictionary of colors to be used for plotting
    df - degrees of freedom of scaled t-distribution, df=1 is usual t-SNE
    
    Outputs:
    Prints grid of plots"""
    
    # Determine dimensions of plot grid
    nrows = len(perp_list) + 1
    ncols = len(data_list)
    
    # Configure axes
    axes = []
    fig = plt.figure(figsize = (16, 3 * nrows))
    
    # Generate plots of starting points (first two columns for high-dimensional)
    for index, dat in enumerate(data_list):
        X, labs = dat
        
        # Check whether original data should be plotted in 3D, and adjust axes accordingly
        if plot_3d:
            axes.append(fig.add_subplot(nrows, ncols, 1 + index, projection = '3d'))
            axes[-1].scatter(xs = X[:, 0], ys = X[:, 1], zs = X[:, 2], edgecolor = None, alpha = 0.8, \
                             c = np.array(list(map(lambda x: cdict[x], labs))))
            axes[-1].set_axis_off()
        else:
            axes.append(fig.add_subplot(nrows, ncols, 1 + index))
            plt.scatter(x = X[:, 0], y = X[:, 1], edgecolor = None, alpha = 0.8, \
                        c = np.array(list(map(lambda x: cdict[x], labs))))
            axes[-1].set_xticklabels([])
            axes[-1].set_yticklabels([])
            axes[-1].xaxis.set_ticks_position('none')
            axes[-1].yaxis.set_ticks_position('none')
        axes[-1].set_title("\n".join(wrap(title_list[index], 35)))        
        
    # Based on function input, generate either perplexity plots of interim iteration plots
        if perp_plot:
            # Generate plots of t-SNE output for different perplexities
            for perp in range(len(perp_list)):
                low_d = tsne(X = X, perplexity = perp_list[perp], verbose = verbose, optim = "fastest", df = df)
                axes.append(fig.add_subplot(nrows, ncols, 1 + index + (perp + 1) * ncols))
                axes[-1].set_title("Perplexity = " + str(perp_list[perp]))
                plt.scatter(x = low_d[-1, :, 0], y = low_d[-1, :, 1], edgecolor = None, alpha = 0.8, \
                            c = np.array(list(map(lambda x: cdict[x], labs))))
                axes[-1].set_xticklabels([])
                axes[-1].set_yticklabels([])
                axes[-1].xaxis.set_ticks_position('none')
                axes[-1].yaxis.set_ticks_position('none')
        else:
            # Generate plots of t-SNE output for different iterations
            low_d = tsne(X = X, perplexity = step_plot_perp, niter = np.max(step_list), verbose = verbose, optim = "fastest", \
                        df = df)
            for step in range(len(step_list)):
                axes.append(fig.add_subplot(nrows, ncols, 1 + index + (step + 1) * ncols))
                axes[-1].set_title("Perplexity = " + str(step_plot_perp) + ", Step = " + str(step_list[step]))
                plt.scatter(x = low_d[step_list[step], :, 0], y = low_d[step_list[step], :, 1], \
                            edgecolor = None, alpha = 0.8,\
                            c = np.array(list(map(lambda x: cdict[x], labs))))
                axes[-1].set_xticklabels([])
                axes[-1].set_yticklabels([])
                axes[-1].xaxis.set_ticks_position('none')
                axes[-1].yaxis.set_ticks_position('none')