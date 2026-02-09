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
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

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


def make_aligned_sign_maps(map_items, animal_dirs, pdf=None):

    uniref = map_items['uniref']
    main_basepath = map_items['composite_basepath']
    img_array = map_items['img_array']

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

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def get_labeled_array(image):
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
    return labeled_array, label_map


def get_region_for_points(labeled_array, points_to_check, label_map):
    rows, cols = labeled_array.shape
    results = []
    for i, (x, y) in enumerate(points_to_check):
        if 0 <= y < rows and 0 <= x < cols:
            region_id = labeled_array[int(y), int(x)]
            results.append([i, y, x, region_id])
        else:
            results.append([i, y, x, -1])

    results = np.array(results)

    return results


def get_cell_data(rdata, key, cond):
    # Map key if necessary
    use_key = key
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    if key in reverse_map:
        use_key = reverse_map[key]

    isrel = None
    mod = None
    peak = None

    # Try key
    isrel_key = f'{use_key}_{cond}_isrel'
    mod_key = f'{use_key}_{cond}_mod'
    
    if isrel_key in rdata:
        isrel = rdata[isrel_key]
        mod = rdata[mod_key]
    
    # Try to get peak data
    pref_key = f'{use_key}_{cond}_pref'
    if pref_key in rdata:
        peak = rdata[pref_key]
    else:
        # Calculate from tuning if available
        tuning_key = f'{use_key}_1dtuning'
        bins_key = f'{use_key}_1dbins'
        
        if tuning_key in rdata and bins_key in rdata:
            tuning = rdata[tuning_key]
            bins = rdata[bins_key]
            # Assuming tuning shape (n_cells, n_bins, 2) where 0=dark, 1=light
            cond_idx = 1 if cond == 'l' else 0
            
            if tuning.ndim == 3 and tuning.shape[2] > cond_idx:
                 peak_indices = np.argmax(tuning[:, :, cond_idx], axis=1)
                 peak = bins[peak_indices]

    return isrel, mod, peak


def get_glm_keys(key):
    # Map topography key to GLM keys
    # Returns (importance_key_suffix, component_r2_key)
    
    map_imp = {
        'theta': 'theta',
        'phi': 'phi',
        'dTheta': 'dTheta',
        'dPhi': 'dPhi',
        'pitch': 'pitch',
        'yaw': 'yaw',
        'roll': 'roll',
        'dPitch': 'gyro_y',
        'dYaw': 'gyro_z',
        'dRoll': 'gyro_x'
    }
    
    map_comp = {
        'theta': 'theta_pos_r2',
        'phi': 'phi_pos_r2',
        'dTheta': 'theta_vel_r2',
        'dPhi': 'phi_vel_r2',
        'pitch': 'pitch_pos_r2',
        'yaw': 'yaw_pos_r2',
        'roll': 'roll_pos_r2',
        'dPitch': 'pitch_vel_r2',
        'dYaw': 'yaw_vel_r2',
        'dRoll': 'roll_vel_r2'
    }
    
    if key in map_imp and key in map_comp:
        return f'full_importance_{map_imp[key]}', map_comp[key]
    return None, None


def plot_variable_summary(pdf, data, key, cond, uniref, img_array, animal_dirs, labeled_array, label_map, glm_map):
    
    # GLM keys
    imp_key, comp_key = get_glm_keys(key)
    glm_cache = {}
    
    # 1. Collect all cell data first
    cells = []
    
    for animal_dir in animal_dirs:
        if animal_dir not in data[key][cond]:
            continue
            
        if 'transform' not in data[key][cond][animal_dir]:
             continue

        for poskey in data[key][cond][animal_dir]['transform'].keys():
            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            # Get cell data handling key mapping
            isrel, mod, peak = get_cell_data(data[key][cond][animal_dir]['messentials'][poskey]['rdata'], key, cond)
            
            if isrel is None:
                continue
            
            transform = data[key][cond][animal_dir]['transform'][poskey]

            for c in range(np.size(isrel, 0)):
                # Retrieve GLM values for this cell if available
                c_imp = np.nan
                c_comp = np.nan
                if glm_map and animal_dir in glm_map:
                     try:
                        pidx = int(poskey.replace('pos', '')) - 1
                        if 0 <= pidx < len(glm_map[animal_dir]):
                            fpath = glm_map[animal_dir][pidx]
                            if fpath not in glm_cache:
                                glm_cache[fpath] = fm2p.read_h5(fpath)
                            gdata = glm_cache[fpath]
                            if imp_key in gdata and c < len(gdata[imp_key]):
                                c_imp = gdata[imp_key][c]
                            if comp_key in gdata and c < len(gdata[comp_key]):
                                c_comp = gdata[comp_key][c]
                     except Exception:
                         pass

                cells.append({
                    'x': transform[c, 2],
                    'y': transform[c, 3],
                    'rel': isrel[c],
                    'mod': mod[c],
                    'peak': peak[c] if peak is not None else np.nan,
                    'imp': c_imp,
                    'comp': c_comp
                })

    if not cells:
        return

    df = pd.DataFrame(cells)
    cond_name = 'Light' if cond == 'l' else 'Dark'

    # 2. Plot for each metric (Modulation, Peak, Importance, Component R2)
    metrics_to_plot = ['mod']
    if not df['peak'].isna().all():
        metrics_to_plot.append('peak')
    if not df['imp'].isna().all():
        metrics_to_plot.append('imp')
    if not df['comp'].isna().all():
        metrics_to_plot.append('comp')

    for metric in metrics_to_plot:
        
        # Determine visualization properties
        if metric == 'mod':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.5)
            label_str = 'Modulation Index'
        elif metric == 'peak':
            # Peak
            if key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                cmap = cm.hsv
                norm = colors.Normalize(vmin=-180, vmax=180)
                label_str = f'{key} Peak (deg)'
            else:
                # Velocities
                limit = np.nanpercentile(np.abs(df['peak']), 95)
                cmap = cm.coolwarm
                norm = colors.Normalize(vmin=-limit, vmax=limit)
                label_str = f'{key} Peak'
        elif metric == 'imp':
            cmap = cm.viridis
            norm = colors.Normalize(vmin=0, vmax=0.1) # Importance usually small positive
            label_str = 'Variable Importance (Shuffle)'
        elif metric == 'comp':
            cmap = cm.magma
            norm = colors.Normalize(vmin=0, vmax=0.5) # R2
            label_str = 'Component Model R2'

        fig = plt.figure(figsize=(6,6), dpi=300)
        gs = GridSpec(5,5)
        ax = fig.add_subplot(gs[1:5, 0:4])

        ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),2), cmap='jet', alpha=0.15)
        ax.imshow(img_array)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Plot non-reliable cells
        unrel = df[df['rel'] == 0]
        ax.plot(unrel['x'], unrel['y'], '.', ms=3, color='gray', alpha=0.15)

        # Plot reliable cells
        if metric == 'peak':
            # Filter for peak map: reliable AND mod > 0.33
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
        else:
            # For others, just reliable
            rel = df[df['rel'] == 1]
            
        if len(rel) > 0:
            sc = ax.scatter(rel['x'], rel['y'], s=3, c=rel[metric], cmap=cmap, norm=norm)

        ax_histx  = fig.add_subplot(gs[0, 0:4], sharex=ax)
        ax_histy  = fig.add_subplot(gs[1:5, 4], sharey=ax)
        ax_cmap = fig.add_subplot(gs[0,4])

        if len(rel) > 0:
            plot_running_median(ax_histx, rel['x'], rel[metric], 7, fb=True)
            plot_running_median(ax_histy, rel['y'], rel[metric], 7, vertical=True, fb=True)

        fig.suptitle(f'{key} {metric.capitalize()} ({cond_name})')

        ax.set_xlim([0,1024])
        ax.set_ylim([0,1024])
        ax.invert_yaxis()

        # Colorbar
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cmap, label=label_str)
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Region Analysis (only for Modulation Index usually, but can do for both)
        if metric == 'mod':
            if len(rel) == 0:
                continue
                
            points_lt = list(zip(rel['x'], rel['y']))
            results = get_region_for_points(labeled_array, points_lt, label_map)
            
            # Add region ID to dataframe
            rel = rel.copy()
            rel['region'] = results[:, 3]

            # Metric per Region
            fig, ax = plt.subplots(1, 1, figsize=(5,3.5), dpi=300)
            ax.hlines(0.33, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            ax.hlines(0.5, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            
            for i in range(6):
                region_vals = rel[rel['region'] == i+2][metric]
                if len(region_vals) > 0:
                    add_scatter_col(ax, i, region_vals)

            ax.set_xticks(np.arange(6), labels=list(label_map.values())[2:8])
            ax.set_ylim([0,0.75])
            ax.set_xlim([-.5,5.5])
            ax.set_ylabel(label_str)
            plt.title(f'{key} {metric} by Region ({cond_name})')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Only plot the region maps grid once (it shows cell locations, not values)
            fig, axs = plt.subplots(2,5,dpi=300, figsize=(8,3))
            axs = axs.flatten()

            # Use all cells for location map
            all_points = list(zip(df['x'], df['y']))
            all_results = get_region_for_points(labeled_array, all_points, label_map)
            
            for i in range(10):
                region_mask = all_results[:, 3] == i+1
                ys = df.loc[region_mask, 'y']
                xs = df.loc[region_mask, 'x']

                axs[i].imshow(labeled_array, cmap='jet')
                axs[i].set_ylim(1022,0)
                axs[i].set_xlim([0,1022])
                if len(ys) > 0:
                    axs[i].plot(xs, ys, 'k.', ms=0.3)
                axs[i].set_title(f'all {label_map[i+1]} cells')
                axs[i].axis('off')
            
            fig.suptitle(f'{key} Cells by Region')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def get_glm_map(root_dir):
    # Find all GLM files
    glm_files = fm2p.find('pytorchGLM_predictions_v04_imurepair.h5', root_dir)
    
    # Organize by animal
    glm_map = {}
    for f in glm_files:
        parts = f.split(os.sep)
        animal = next((p for p in parts if p.startswith('DMM')), None)
        if animal:
            if animal not in glm_map:
                glm_map[animal] = []
            glm_map[animal].append(f)
            
    for animal in glm_map:
        glm_map[animal].sort()
    return glm_map


def main():

    uniref = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')
    data = fm2p.read_h5('/home/dylan/Fast1/freely_moving_data/pooled_datasets/pooled_260127.h5')
    img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)
    composite_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'
    
    # GLM files
    cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/'
    glm_map = get_glm_map(cohort_dir)

    # Rename keys if necessary
    key_map = {'gyro_x': 'dRoll', 'gyro_y': 'dPitch', 'gyro_z': 'dYaw'}
    for old, new in key_map.items():
        if old in data:
            data[new] = data.pop(old)

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    conditions = ['l', 'd']
    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']

    map_items = {
        'pooled_data': data,
        'uniref': uniref,
        'img_array': img_array,
        'composite_basepath': composite_basepath
    }

    labeled_array, label_map = get_labeled_array(img_array[:,:,0].clip(max=1))

    with PdfPages('topography_summary.pdf') as pdf:
        
        make_aligned_sign_maps(map_items, animal_dirs, pdf=pdf)

        for key in tqdm(variables, desc="Processing variables"):
            if key not in data:
                print(f"Variable {key} not found in data, skipping.")
                continue
                
            for cond in conditions:
                if cond not in data[key]:
                    continue
                
                plot_variable_summary(
                    pdf, data, key, cond, uniref, img_array, 
                    animal_dirs, labeled_array, label_map, glm_map
                )


if __name__ == '__main__':

    main()
