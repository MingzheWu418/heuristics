import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib import pyplot as plt
from envs_numpy import V, u
import torch
import imageio
import os
from copy import deepcopy

# the function that I'm going to plot
def z_func(X, Y, users):
    Z = np.zeros_like(X)
    np_users = users
    zero_pad = np.zeros_like(np_users[:-1,:])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s = np.vstack([np.array([X[i,j], Y[i,j]]),zero_pad])
            Z[i, j] = V(s, np_users)
    # print(Z)
    return Z
    
def fit_xyticks(trace):
    return (trace+2)*20

def plot_S_trace(S_history, users, directory="", file_name="", n_step=100):
    x = np.arange(-2.0,2.0,0.05)
    y = np.arange(-2.0,2.0,0.05)
    X, Y = meshgrid(x, y) # grid of point
    Z = z_func(X, Y, users) # evaluation of the function on the grid

    colors = ['aqua', 'black', 'darkblue', 'darkgreen', 'darkviolet', 'gold']
    plt.figure(figsize = (15,15))
    im = plt.imshow(Z, cmap=cm.RdBu) # drawing the function
    # print(S_history.shape)
    for i in range(S_history.shape[1]):
        # print(type(S_history))
        # print(n_step)
        # print(i)
        # print(S_history[:50,0,0])
        plt.scatter(fit_xyticks(S_history[:n_step,i,0]),fit_xyticks(S_history[:n_step,i,1]), color=colors[i], marker='.')

    # adding the Contour lines with labels
    cset = contour(Z, np.arange(3.1,4.3,0.1), linewidths=1, cmap=cm.Set2)
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im) # adding the colobar on the right
    # latex fashion title
    title('V(s)')
    plt.savefig(directory + file_name)
    show()
    plt.clf()
    plt.close()

def plot_trace_utility(S_history, users, action, directory="", file_name="", n_step=100):

    for step in range(S_history.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(15,15))
        fig.tight_layout()

        x = np.arange(-2.0,2.0,0.05)
        y = np.arange(-2.0,2.0,0.05)
        X, Y = meshgrid(x, y) # grid of point
        Z = z_func(X, Y, users) # evaluation of the function on the grid

        colors = ['aqua', 'black', 'darkblue', 'darkgreen', 'darkviolet', 'gold']
        # plt.figure(figsize = (15,15))
        im = ax[0].imshow(Z, cmap=cm.RdBu) # drawing the function
        # print(S_history.shape)
        ax[1].set_ylim([0.0, 0.25])
        
        for i in range(S_history.shape[1]):
            ax[0].scatter(fit_xyticks(S_history[step,i,0]),fit_xyticks(S_history[step,i,1]), color=colors[i], marker='.')
        utilities, _ = u(S_history[step], users, tau=1.0, top_k=1, user_weight=np.array([1,1,1,1,1]))
        # ax[1].bar(np.arange(0, 5, 1), utilities.cpu().detach().numpy(), color=colors)

            # adding the Contour lines with labels
        cset = ax[0].contour(Z, np.arange(3.1,4.3,0.1), linewidths=1, cmap=cm.Set2)
        clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        # colorbar(im) # adding the colobar on the right

        #create subplots
        # ax[0].plot(x, y, color='red')
        # ax[1].plot(x, y, color='blue')
        # ax[1, 0].plot(x, y, color='green')
        # ax[1, 1].plot(x, y, color='purple')
        # title('V(s)')
        plt.savefig(directory + file_name + "_" + str(step))
        # show()
        plt.clf()
        plt.close()
    

    gif_fig_list = []
    for n_iter in range(n_step):
        gif_fig_list.append(directory + file_name + "_" + str(n_iter) + ".png")

    ims = [imageio.imread(f) for f in gif_fig_list]
    imageio.mimwrite(directory + file_name + ".gif", ims, fps=5)
    #using listdir() method to list the files of the folder
    test = os.listdir(directory)
    #taking a loop to remove all the images
    #using ".png" extension to remove only png images
    #using os.remove() method to remove the files
    for images in test:
        if images.endswith(".png"):
            os.remove(os.path.join(directory, images))

def plot_individual_utilities(strategies, users, action=np.array([1,1,1,1,1]), directory="", file_name=""):
    x = np.arange(-2.0,2.0,0.05)
    y = np.arange(-2.0,2.0,0.05)
    X, Y = meshgrid(x, y) # grid of point

    for i in range(strategies.shape[0]):
        Z = np.zeros_like(X)
        for x_coor in range(X.shape[0]):
            for y_coor in range(X.shape[1]):
                curr_strategy = np.array([X[x_coor, y_coor], Y[x_coor, y_coor]])
                # print(curr_strategy)
                temp_strategies = deepcopy(strategies)
                temp_strategies[i] = curr_strategy
                # print(temp_strategies)
                # print(torch.Tensor(temp_strategies).cuda())
                utilities, _ = u(temp_strategies, users, user_weight=action)
                Z[x_coor, y_coor] = utilities[i]

        colors = ['aqua', 'black', 'darkblue', 'darkgreen', 'darkviolet', 'gold']
        plt.figure(figsize = (15,15))
        im = plt.imshow(Z, cmap=cm.RdBu) # drawing the function
        # print(S_history.shape)
        for j in range(strategies.shape[0]):
            # print(type(S_history))
            # print(n_step)
            # print(i)
            # print(S_history[:50,0,0])
            plt.scatter(fit_xyticks(strategies[j,0]),fit_xyticks(strategies[j,1]), color=colors[j], marker='.')

        # adding the Contour lines with labels
        cset = contour(Z, np.arange(0.0,3.0,0.1), linewidths=1, cmap=cm.Set2)
        clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        colorbar(im) # adding the colobar on the right
        # latex fashion title
        title('V(s)')
        plt.savefig(directory + file_name + "_player_" + str(i))
        show()
        plt.clf()
        plt.close()