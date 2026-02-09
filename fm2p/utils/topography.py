# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.gridspec import GridSpec
from skimage.measure import label, regionprops
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

import fm2p


def plot_running_median(ax, x, y, n_bins=7, vertical=False, fb=True):

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    if not vertical:
        ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                bin_means,
                '-', color='k')
        if fb:
            ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)
    
    elif vertical:
        ax.plot(bin_means,
                bin_edges[:-1] + (np.median(np.diff(bins))/2),
                '-', color='k')
        
        if fb:
            ax.fill_betweenx(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)


def shift_image(image, dx, dy):
    translation_vector = (dx, dy) 
    transform = skimage.transform.AffineTransform(
        translation=translation_vector
    )
    shifted_image = skimage.transform.warp(
        image,
        transform,
        mode='constant',
        preserve_range=True
    )
    return shifted_image


def add_scatter_col(ax, pos, vals):

    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/2,
        vals,
        s=2, c='k'
    )
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='r')

    stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals)-stderr, np.nanmean(vals)+stderr, color='r')


def make_aligned_sign_maps(map_items):

    uniref = map_items['uniref']
    main_basepath = map_items['composite_basepath']
    img_array = map_items['img_array']

    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']

    fig, ax = plt.subplots(1, 1, figsize=(2,2), dpi=300)

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),12), cmap='gray', alpha=0.3)

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)
        if animal_dir != 'DMM056':
            transform_g2u = fm2p.read_h5(fm2p.find('aligned_composite_*.h5', basepath, MR=True))
            messentials = fm2p.read_h5(fm2p.find('*_merged_essentials_v6.h5', basepath, MR=True))
        else:
            continue

        k = list(transform_g2u.keys())[0]
        x_shift = transform_g2u[k][0][2] - transform_g2u[k][0][0]
        y_shift = transform_g2u[k][0][3] - transform_g2u[k][0][1]

        shifted_sign_map = shift_image(messentials['sign_map'], -x_shift, -y_shift)

        ax.imshow(gaussian_filter(shifted_sign_map, 10), alpha=0.3, cmap='jet')

    ax.set_title('Aligned sign maps')

    ax.imshow(img_array)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_ylim([1022, 0])
    ax.set_xlim([0,1022])

    fig.tight_layout()


def process_shapes_with_border_logic(image, points_to_check):

    label_map = {
        0: 'boundary',
        1: 'outside',
        2: 'RL',
        3: 'RLL',
        4: 'MMA',
        5: 'AM',
        6: 'PM',
        7: 'V1',
        8: 'MMP',
        9: 'AL',
        10: 'LM',
        11: 'P'
    }

    labeled_array = label(image, connectivity=1)

    border_labels = set()
    rows, cols = labeled_array.shape

    border_labels.update(np.unique(labeled_array[0, :]))      # Top row
    border_labels.update(np.unique(labeled_array[rows-1, :])) # Bottom row
    border_labels.update(np.unique(labeled_array[:, 0]))      # Left col
    border_labels.update(np.unique(labeled_array[:, cols-1])) # Right col

    all_labels = np.unique(labeled_array)
    all_labels = all_labels[all_labels != 0]

    results = []
    for i, (x, y) in enumerate(points_to_check):
        if 0 <= y < rows and 0 <= x < cols:
            region_id = labeled_array[int(y), int(x)]
            category = label_map.get(region_id, "Unknown")
            results.append([i, y, x, region_id])
        else:
            results.append([i, y, x, -1])

    results = np.array(results)

    return results, labeled_array, label_map


def main():

    uniref = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')

    data = fm2p.read_h5('/home/dylan/Fast1/freely_moving_data/pooled_datasets/pooled_260127.h5')

    img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)

    composite_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    map_items = {
        'pooled_data': data,
        'uniref': uniref,
        'img_array': img_array,
        'composite_basepath': composite_basepath
    }

    make_aligned_sign_maps(map_items)

    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']

    keys = ['theta']
    conds = ['l','d']
    hist_data = []
    ii = 0
    for key in keys:
        for cond in conds:

            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.5)

            h_hist_data = []
            v_hist_data = []

            fig = plt.figure(figsize=(6,6), dpi=300)

            gs = GridSpec(5,5)

            ax = fig.add_subplot(gs[1:5, 0:4])

            ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),2), cmap='jet', alpha=0.15)

            ax.imshow(img_array)
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)

            for animal_dir in animal_dirs:
                
                kca_data = data[key][cond][animal_dir]['messentials']

                for poskey in data[key][cond][animal_dir]['transform'].keys():

                    if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                        continue

                    for c in range(np.size(data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key,cond)], 0)):
                        cellx = data[key][cond][animal_dir]['transform'][poskey][c,2] # was 2
                        celly = data[key][cond][animal_dir]['transform'][poskey][c,3] # was 3
                        cellrel = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key, cond)][c]

                        if cellrel:
                            cellmod = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]
                            ax.plot(cellx, celly, '.', ms=3, color=cmap(norm(cellmod)))

                            h_hist_data.append([cellx, cellmod, ii])
                            v_hist_data.append([celly, cellmod, ii])


                        elif not cellrel:
                            ax.plot(cellx, celly, '.', ms=3, color='gray', alpha=0.15)

                    ii += 1

            ax_histx  = fig.add_subplot(gs[0, 0:4], sharex=ax)
            ax_histy  = fig.add_subplot(gs[1:5, 4], sharey=ax)

            ax_cmap = fig.add_subplot(gs[0,4])

            h_hist_data = np.array(h_hist_data)
            v_hist_data = np.array(v_hist_data)

            plot_running_median(ax_histx, h_hist_data[:,0], h_hist_data[:,1], 7, fb=True)
            plot_running_median(ax_histy, v_hist_data[:,0], v_hist_data[:,1], 7, vertical=True, fb=True)

            hist_data.append([h_hist_data, v_hist_data])

            fig.suptitle('{} ({})'.format(key, cond))

            ax.set_xlim([0,1024])
            ax.set_ylim([0,1024])

            fig1, ax1 = plt.subplots(1,1)
            im = ax1.imshow(cmap(norm(np.linspace(-0.5, 1.5, 100))), cmap=cm.plasma, vmin=0, vmax=0.5)
            ax1.axis('off')
            ax_cmap.axis('off')
            plt.colorbar(im, ax=ax_cmap, label='MI')

            ax.invert_yaxis()

            fig.tight_layout()

            points_lt = []
            for i in range(np.size(h_hist_data, 0)):
                points_lt.append((h_hist_data[i, 0], v_hist_data[i, 0]))

            results, labeled_array, label_map = process_shapes_with_border_logic(img_array[:,:,0].clip(max=1), points_lt)

            plt.figure(dpi=300, figsize=(4,4))
            plt.imshow(labeled_array, cmap='jet')
            plt.ylim(1022,0)
            plt.xlim([0,1022])
            plt.colorbar()
            plt.tight_layout()

            groups = pd.DataFrame([])
            for i in range(6):
                inds = results[:,0][results[:,3] == i+2]
                vals = h_hist_data[inds.astype(int),1]
                temp_ = pd.concat([pd.Series(np.ones(np.size(vals)) * i), pd.Series(vals)], axis=1)
                groups = pd.concat([groups, temp_])
            groups.columns = ['Group','Value']

            fig, ax = plt.subplots(1, 1, figsize=(5,3.5), dpi=300)
            ax.hlines(0.33, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            ax.hlines(0.5, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            for i in range(6):
                inds_lt = results[:,0][results[:,3] == i+2]
                vals_lt = h_hist_data[inds_lt.astype(int),1]
                add_scatter_col(ax, i, vals_lt)

            ax.set_xticks(np.arange(6), labels=list(label_map.values())[2:8])
            ax.set_ylim([0,0.75])
            ax.set_xlim([-.5,5.5])
            ax.set_ylabel('modulation index')
            plt.title('phi reliable cells only')
            fig.tight_layout()

            fig, axs = plt.subplots(2,5,dpi=300, figsize=(8,3))
            axs = axs.flatten()

            for i in range(10):

                ys = results[:,1][results[:,3] == i+1]
                xs = results[:,2][results[:,3] == i+1]

                axs[i].imshow(labeled_array, cmap='jet')
                axs[i].set_ylim(1022,0)
                axs[i].set_xlim([0,1022])
                for c in range(np.size(ys)):
                    axs[i].plot(xs[c], ys[c], 'k.', ms=0.3)
                axs[i].set_title('all {} cells'.format(label_map[i+1]))
            fig.tight_layout()


    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']
    main_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'



    # img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)
    # fig, ax = plt.subplots()
    # ax.imshow(img_array)
    # fig.patch.set_alpha(0)
    # ax.patch.set_alpha(0)

    fig, ax = plt.subplots(1, 1, figsize=(3,3), dpi=300)

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),12), cmap='gray', alpha=0.3)

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)
        if animal_dir != 'DMM056':
            transform_g2u = fm2p.read_h5(fm2p.find('aligned_composite_*.h5', basepath, MR=True))
            messentials = fm2p.read_h5(fm2p.find('*_merged_essentials_v6.h5', basepath, MR=True))
        else:
            continue

        k = list(transform_g2u.keys())[0]
        x_shift = transform_g2u[k][0][2] - transform_g2u[k][0][0]
        y_shift = transform_g2u[k][0][3] - transform_g2u[k][0][1]

        shifted_sign_map = shift_image(messentials['sign_map'], -x_shift, -y_shift)

        ax.imshow(gaussian_filter(shifted_sign_map, 10), alpha=0.3, cmap='jet')

    ax.set_title('phi-reliable V1 cells over aligned sign maps')

    ax.imshow(img_array)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_ylim([1022, 0])
    ax.set_xlim([0,1022])

    cmap = cm.plasma
    norm = colors.Normalize(vmin=0, vmax=0.5)

    for c in range(np.size(usedata,0)):
        ax.plot(usedata[c,2], usedata[c,1], '.', ms=2, color=cmap(norm(usedata[c,3])))

    fig.tight_layout()






    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']
    # main_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    ii = 0

    key = 'phi'
    cond = 'l'
    cmap = cm.coolwarm
    norm = colors.Normalize(vmin=-15, vmax=15)

    h_hist_data = []
    v_hist_data = []

    fig = plt.figure(figsize=(6,6), dpi=300)

    gs = GridSpec(5,5)

    ax = fig.add_subplot(gs[1:5, 0:4])

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),2), cmap='jet', alpha=0.15)

    ax.imshow(img_array)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for animal_dir in animal_dirs:
        
        kca_data = data[key][cond][animal_dir]['messentials']

        for poskey in data[key][cond][animal_dir]['transform'].keys():

            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            for c in range(np.size(data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key,cond)], 0)):
                cellx = data[key][cond][animal_dir]['transform'][poskey][c,2] # was 2
                celly = data[key][cond][animal_dir]['transform'][poskey][c,3] # was 3
                cellrel = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key, cond)][c]
                cellmod = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]

                if cellrel and (cellmod > 0.33):
                    if cond == 'l':
                        condint = 1
                    elif cond == 'd':
                        condint = 0
                    cellpeak = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_1dbins'.format(key)][np.argmax(data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_1dtuning'.format(key)][c,:,condint])]
                    
                    ax.plot(cellx, celly, '.', ms=3, color=cmap(norm(cellpeak)))

                    h_hist_data.append([cellx, cellpeak, ii])
                    v_hist_data.append([celly, cellpeak, ii])


                elif not cellrel:
                    ax.plot(cellx, celly, '.', ms=3, color='gray', alpha=0.15)

            ii += 1

    ax_histx  = fig.add_subplot(gs[0, 0:4], sharex=ax)
    ax_histy  = fig.add_subplot(gs[1:5, 4], sharey=ax)

    ax_cmap = fig.add_subplot(gs[0,4])

    h_hist_data = np.array(h_hist_data)
    v_hist_data = np.array(v_hist_data)

    plot_running_median(ax_histx, h_hist_data[:,0], h_hist_data[:,1], 7, fb=True)
    plot_running_median(ax_histy, v_hist_data[:,0], v_hist_data[:,1], 7, vertical=True, fb=True)

    fig.suptitle('{} ({})'.format(key, cond))

    ax.set_xlim([0,1024])
    ax.set_ylim([0,1024])

    fig1, ax1 = plt.subplots(1,1)
    im = ax1.imshow(cmap(norm(np.linspace(-0.5, 1.5, 100))), cmap=cm.coolwarm, vmin=-15, vmax=15)
    ax1.axis('off')
    ax_cmap.axis('off')
    plt.colorbar(im, ax=ax_cmap, label='theta pref')

    ax.invert_yaxis()

    fig.tight_layout()



    points = []

    for i in range(np.size(h_hist_data,0)):
        points.append((h_hist_data[i,0], v_hist_data[i,0]))

    results, labeled_array, label_map = process_shapes_with_border_logic(img_array[:,:,0].clip(max=1), points)

    area_num = 7

    usedata = results[results[:,3] == area_num, :].copy() # 7 is V1

    usedata[:,3] = h_hist_data[results[:,3] == area_num, 1].copy()

    recid = h_hist_data[results[:,3] == area_num, 2].copy()






    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']
    main_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    # img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)
    # fig, ax = plt.subplots()
    # ax.imshow(img_array)
    # fig.patch.set_alpha(0)
    # ax.patch.set_alpha(0)

    fig, ax = plt.subplots(1, 1, figsize=(3,3), dpi=300)

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),12), cmap='gray', alpha=0.3)

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)
        if animal_dir != 'DMM056':
            transform_g2u = fm2p.read_h5(fm2p.find('aligned_composite_*.h5', basepath, MR=True))
            messentials = fm2p.read_h5(fm2p.find('*_merged_essentials_v6.h5', basepath, MR=True))
        else:
            continue

        k = list(transform_g2u.keys())[0]
        x_shift = transform_g2u[k][0][2] - transform_g2u[k][0][0]
        y_shift = transform_g2u[k][0][3] - transform_g2u[k][0][1]

        shifted_sign_map = shift_image(messentials['sign_map'], -x_shift, -y_shift)

        ax.imshow(gaussian_filter(shifted_sign_map, 10), alpha=0.3, cmap='jet')

    ax.set_title('phi-reliable V1 cells over aligned sign maps')

    ax.imshow(img_array)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_ylim([1022, 0])
    ax.set_xlim([0,1022])

    cmap = cm.coolwarm
    norm = colors.Normalize(vmin=-15, vmax=15)

    for c in range(np.size(usedata,0)):
        ax.plot(usedata[c,2], usedata[c,1], '.', ms=2, color=cmap(norm(usedata[c,3])))

    fig.tight_layout()



if __name__ == '__main__':
    main()

