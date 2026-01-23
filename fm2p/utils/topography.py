# -*- coding: utf-8 -*-



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import fm2p


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():

    json_path = r'D:\freely_moving_data\V1PPC_cohort01\pooled\DMM042.json'
    recdict = read_json(json_path)

    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm1 = colors.Normalize(vmin=0, vmax=0.6)
    cmap = cm.viridis
    cmap1 = cm.cool

    fig1, ax1 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig3, ax3 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig4, ax4 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig5, ax5 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig6, ax6 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig7, ax7 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig8, ax8 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig9, ax9 = plt.subplots(1, 1, figsize=(8,8), dpi=300)
    fig10, ax10 = plt.subplots(1, 1, figsize=(8,8), dpi=300)

    all_pdata = []
    all_rdata = []
    all_pos = []

    full_map = np.zeros([512*5, 512*5]) * np.nan
    row = 0
    col = 0
    for pos in range(1,26):
        pos_str = 'pos{:02d}'.format(pos)
        if pos_str not in list(recdict.keys()):
            if (pos%5)==0: # if you're at the end of a row
                col = 0
                row += 1
            else:
                col += 1
            continue
        pdata = fm2p.read_h5(recdict[pos_str]['preproc'])
        rdata = fm2p.read_h5(recdict[pos_str]['revcorr'])

        all_pdata.append(pdata)
        all_rdata.append(rdata)
        all_pos.append((row, col))


        singlemap = np.fliplr(pdata['twop_ref_img'])

        full_map[row*512 : (row*512)+512, col*512 : (col*512)+512] = singlemap

        for k in pdata['cell_x_pix'].keys():
            cellx = np.median(512 - pdata['cell_x_pix'][k]) + col*512
            celly = np.median(pdata['cell_y_pix'][k]) + row*512
            ax1.plot(cellx, celly, 'r.', ms=1)

        for k in pdata['cell_x_pix'].keys():
            cellx = np.median(512 - pdata['cell_x_pix'][k]) + col*512
            celly = np.median(pdata['cell_y_pix'][k]) + row*512

            if 'light' in rdata.keys():

                cm2 = ax2.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['light']['theta']['is_reliable'][int(k)])))
                cm3 = ax3.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['light']['phi']['is_reliable'][int(k)])))
                cm4 = ax4.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['dark']['theta']['is_reliable'][int(k)])))
                cm5 = ax5.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['dark']['phi']['is_reliable'][int(k)])))
                
                if rdata['light']['theta']['is_reliable'][int(k)]:
                    cm6 = ax6.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['light']['theta']['modulation'][int(k)])))
                if rdata['light']['phi']['is_reliable'][int(k)]:
                    cm7 = ax7.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['light']['phi']['modulation'][int(k)])))
                if rdata['dark']['theta']['is_reliable'][int(k)]:
                    cm8 = ax8.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['dark']['theta']['modulation'][int(k)])))
                if rdata['dark']['phi']['is_reliable'][int(k)]:
                    cm9 = ax9.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['dark']['phi']['modulation'][int(k)])))

            else:

                cm2 = ax2.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['theta']['is_reliable'][int(k)])))
                cm3 = ax3.plot(cellx, celly, '.', ms=1, color=cmap(norm(rdata['phi']['is_reliable'][int(k)])))
                
                if rdata['theta']['is_reliable'][int(k)]:
                    cm6 = ax6.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['theta']['modulation'][int(k)])))
                if rdata['phi']['is_reliable'][int(k)]:
                    cm7 = ax7.plot(cellx, celly, '.', ms=1, color=cmap1(norm1(rdata['phi']['modulation'][int(k)])))
        col += 1

        if (pos%5)==0:
            col = 0
            row += 1

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]:
        ax.imshow(full_map, cmap='gray')

    ax1.set_title('all cells')
    ax2.set_title('light theta reliability')/home/dylan/Storage4/V1PPC_cohort02/251016_DMM_DMM061_pos18/fm1
    ax3.set_title('light phi reliability')
    ax4.set_title('dark theta reliability')
    ax5.set_title('dark phi reliability')
    ax6.set_title('light theta modulation')
    ax7.set_title('light phi modulation')
    ax8.set_title('dark theta modulation')
    ax9.set_title('dark phi modulation')

    for fig in [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10]:
        
        fig.tight_layout()/home/dylan/Storage4/V1PPC_cohort02/251016_DMM_DMM061_pos18/fm1