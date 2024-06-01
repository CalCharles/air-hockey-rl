import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectories(target, tidx, paddle, puck):
    plt.clf()
    fig, ax = plt.subplots()
    # import ipdb;ipdb.set_trace()
    puck_abs = puck

    ax.plot([x[0] for x in paddle], [x[1] for x in paddle], label='Paddle',color='green')
    
    ax.plot(puck_abs[:,0], puck_abs[:,1], label='Puck',color='blue')
    # ellipse1 = Ellipse((0.307-1, -0.107), width=2*0.04, height=2*0.05, edgecolor='blue', facecolor='none')
    # ellipse2 = Ellipse((0.307-1, 0.215), width=2*0.04, height=2*0.05, edgecolor='blue', facecolor='none')
    # ellipse3 = Ellipse((0.570-1, -0.215), width=2*0.04, height=2*0.05, edgecolor='blue', facecolor='none')
    # ellipse4 = Ellipse((0.570-1, 0.107), width=2*0.04, height=2*0.05, edgecolor='blue', facecolor='none')
    # ax.plot(0.219-1, 0.323, marker='*', color='red', markersize=15, label='Goal')
    
    
    
    # # Add the ellipse to the plot
    # ax.add_patch(ellipse1)
    # ax.add_patch(ellipse2)
    # ax.add_patch(ellipse3)
    # ax.add_patch(ellipse4)
    # ax.set_aspect('equal')
    
    # # Set grid, labels and title for better visualization
    ax.grid(True)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_xlim([0, 0.85])
    ax.set_ylim([-0.45, 0.45])
    plt.savefig(os.path.join(target, 'expert_{}.png'.format(str(tidx))))





