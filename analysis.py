import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

def load_lossed(path="data/losses.txt"):
    with open(path, "r",encoding="utf-8") as fi:
        losses = json.load(fi)
    return losses

def show_losses():
    # 设置label的字体
    plt.rcParams['axes.titlesize']=14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    
    losses = load_lossed()
    y1 = np.array(losses)
    x1 = np.array(range(len(losses)))
    
    fig, ax1 = plt.subplots(1, 1)
    
    ax1.set_title("")
    # ax1.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6])
    ax1.set_ylim((0.0, 3.0))
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Training steps")
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(bottom=True, top=False, left=False, right=False)
    ax1.patch.set_facecolor("whitesmoke") 
    # 第一条折线
    ax1.plot(x1, y1, 
            color="#72687E",# 颜色
            linestyle='-',  # 样式
            linewidth=1.0,  # 宽度
            alpha=1.0,  # 透明度  
            )
    
    ax1.grid(axis='y') 
    # ax1.legend()
    
    

    plt.show()

show_losses()