# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import label
import scipy.stats
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import skimage.transform

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
        3: 'AM',
        4: 'PM',
        5: 'V1',
        7: 'AL',
        8: 'LM',
        9: 'P'
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

    use_key = key
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    if key in reverse_map:
        use_key = reverse_map[key]

    isrel = None
    mod = None
    peak = None

    isrel_key = f'{use_key}_{cond}_isrel'
    mod_key = f'{use_key}_{cond}_mod'
    
    if isrel_key in rdata:
        isrel = rdata[isrel_key]
        mod = rdata[mod_key]
    
    pref_key = f'{use_key}_{cond}_pref'
    if pref_key in rdata:
        peak = rdata[pref_key]
    else:
        tuning_key = f'{use_key}_1dtuning'
        bins_key = f'{use_key}_1dbins'
        
        if tuning_key in rdata and bins_key in rdata:
            tuning = rdata[tuning_key]
            bins = rdata[bins_key]

            cond_idx = 1 if cond == 'l' else 0
            
            if tuning.ndim == 3 and tuning.shape[2] > cond_idx:
                 peak_indices = np.argmax(tuning[:, :, cond_idx], axis=1)
                 peak = bins[peak_indices]

    return isrel, mod, peak


def get_glm_keys(key):

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
    
    if key in map_imp:
        return f'full_importance_{map_imp[key]}'
    return None


def create_smoothed_map(x, y, values, shape=(1024, 1024), sigma=10):
    
    if len(values) < 4:
        return np.full(shape, np.nan)

    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.column_stack((y, x))
    
    smoothed = griddata(points, values, (grid_y, grid_x), method='linear')
    
    mask_nan = np.isnan(smoothed)
    if np.any(mask_nan):
        smoothed_nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
        smoothed[mask_nan] = smoothed_nearest[mask_nan]
    
    smoothed = gaussian_filter(smoothed, sigma=sigma, mode='nearest')
    
    return smoothed


def plot_variable_summary(pdf, data, key, cond, uniref, img_array, animal_dirs, labeled_array, label_map):
    
    imp_key = get_glm_keys(key)
    
    cells = []
    
    for animal_dir in animal_dirs:
        if animal_dir not in data.keys():
            continue
            
        if 'transform' not in data[animal_dir].keys():
             continue

        for poskey in data[animal_dir]['transform'].keys():
            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            isrel, mod, peak = get_cell_data(data[animal_dir]['messentials'][poskey]['rdata'], key, cond)
            
            if isrel is None:
                continue
            
            transform = data[animal_dir]['transform'][poskey]

            model_data = data[animal_dir]['messentials'][poskey].get('model', {})

            for c in range(np.size(isrel, 0)):
                c_imp = np.nan
                
                if isinstance(model_data, dict):
                    if imp_key and imp_key in model_data and c < len(model_data[imp_key]):
                        c_imp = model_data[imp_key][c]

                cells.append({
                    'x': transform[c, 2],
                    'y': transform[c, 3],
                    'rel': isrel[c],
                    'mod': mod[c],
                    'peak': peak[c] if peak is not None else np.nan,
                    'imp': c_imp # feature importance
                })

    if not cells:
        return

    df = pd.DataFrame(cells)
    cond_name = 'Light' if cond == 'l' else 'Dark'

    metrics_to_plot = ['mod']
    metrics_to_plot.append('peak')
    metrics_to_plot.append('imp')

    for metric in metrics_to_plot:
        
        if metric == 'mod':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.5)
            label_str = 'Modulation Index'
        elif metric == 'peak':
            if key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                cmap = cm.coolwarm
                norm = colors.Normalize(vmin=-15, vmax=15)
                label_str = f'{key} Peak (deg)'
            else:
                limit = np.nanpercentile(np.abs(df['peak']), 95)
                cmap = cm.plasma
                norm = colors.Normalize(vmin=-limit, vmax=limit)
                label_str = f'{key} Peak'
        elif metric == 'imp':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.1)
            label_str = 'Variable Importance (Shuffle)'

        if metric == 'peak':
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
        else:
            rel = df[df['rel'] == 1]

        if len(rel) == 0:
            continue
            
        points_lt = list(zip(rel['x'], rel['y']))
        results = get_region_for_points(labeled_array, points_lt, label_map)
        
        rel = rel.copy()
        rel['region'] = results[:, 3]

        fig, ax = plt.subplots(1, 1, figsize=(5,3.5), dpi=300)
        if metric == 'mod':
            ax.hlines(0.33, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            ax.hlines(0.5, -1, 7, color='tab:grey', ls='--', alpha=0.56)
        
        for i in range(4):
            region_vals = rel[rel['region'] == i+2][metric]
            if len(region_vals) > 0:
                add_scatter_col(ax, i, region_vals)

        ax.set_xticks(np.arange(4), labels=list(label_map.values())[2:6])
        if metric == 'mod':
            ax.set_ylim([0,0.75])
        ax.set_xlim([-.5,3.5])
        ax.set_ylabel(label_str)
        plt.title(f'{key} {metric} by Region ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,6))
        axs = axs.flatten()

        for i in range(4):
            region_id = [2,3,4,5][i]
            region_name = label_map.get(region_id, f'Region {region_id}')
            
            region_mask = (labeled_array == region_id)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1) # Boundaries
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            
            region_cells = rel[rel['region'] == region_id]
            if len(region_cells) > 0:
                axs[i].scatter(region_cells['x'], region_cells['y'], s=2, c=region_cells[metric], cmap=cmap, norm=norm)
            
            axs[i].set_title(f'{region_name}')
            axs[i].axis('off')
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])

        fig.suptitle(f'{key} {metric} Map by Region ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,6))
        axs = axs.flatten()

        for i in range(4):
            region_id = [2,3,4,5][i]
            region_name = label_map.get(region_id, f'Region {region_id}')
            
            region_mask = (labeled_array == region_id)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            
            region_cells = rel[rel['region'] == region_id]
            if len(region_cells) > 0:
                smoothed = create_smoothed_map(region_cells['x'].values, region_cells['y'].values, region_cells[metric].values, shape=labeled_array.shape)
                smoothed[~region_mask] = np.nan
                axs[i].imshow(smoothed, cmap=cmap, norm=norm)
            
            axs[i].set_title(f'{region_name}')
            axs[i].axis('off')
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{key} {metric} Smoothed Map by Region ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_model_performance(pdf, data, uniref, img_array, animal_dirs, labeled_array, label_map):
    
    models = ['full', 'velocity_only', 'position_only', 'eyes_only', 'head_only']
    metrics = ['r2', 'corrs']
    
    for model in models:
        for metric in metrics:
            
            cells = []
            for animal_dir in animal_dirs:
                if animal_dir not in data: continue
                if 'transform' not in data[animal_dir]: continue
                
                for poskey in data[animal_dir]['transform']:
                    transform = data[animal_dir]['transform'][poskey]
                    model_data = data[animal_dir]['messentials'][poskey].get('model', {})
                    
                    m_key = f'{model}_{metric}'
                    if m_key not in model_data:
                        continue
                        
                    vals = model_data[m_key]
                    
                    for c in range(len(vals)):
                        cells.append({
                            'x': transform[c, 2],
                            'y': transform[c, 3],
                            'val': vals[c]
                        })
            
            if not cells:
                print(f"No data found for {model} {metric}")
                continue
                
            df = pd.DataFrame(cells)
            
            points_lt = list(zip(df['x'], df['y']))
            results = get_region_for_points(labeled_array, points_lt, label_map)
            df['region'] = results[:, 3]
            
            cmap = cm.plasma
            if metric == 'r2':
                norm = colors.Normalize(vmin=0, vmax=0.5)
                label_str = f'{model} R²'
            else:
                norm = colors.Normalize(vmin=0, vmax=0.8)
                label_str = f'{model} Correlation'

            fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,8))
            axs = axs.flatten()

            for i in range(4):
                region_id = [2,3,4,5][i]
                region_name = label_map.get(region_id, f'Region {region_id}')

                region_mask = (labeled_array == region_id)
                axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
                axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
                axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
                
                region_cells = df[df['region'] == region_id]
                if len(region_cells) > 0:
                    axs[i].scatter(region_cells['x'], region_cells['y'], s=2, c=region_cells['val'], cmap=cmap, norm=norm)
                
                axs[i].set_title(f'{region_name}')
                axs[i].axis('off')
                axs[i].set_ylim([labeled_array.shape[0], 0])
                axs[i].set_xlim([0, labeled_array.shape[1]])
            
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=label_str, fraction=0.05, shrink=0.6)

            fig.suptitle(f'{label_str} Map by Region')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():

    uniref = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')
    data = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260208.h5')
    img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)
    # composite_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    conditions = ['l']
    animal_dirs = ['DMM037','DMM041', 'DMM042','DMM056', 'DMM061']
    labeled_array, label_map = get_labeled_array(img_array[:,:,0].clip(max=1))

    with PdfPages('topography_summary_v06.pdf') as pdf:
    

        for key in tqdm(variables, desc="Processing variables"):
            for cond in conditions:
                plot_variable_summary(
                    pdf, data, key, cond, uniref, img_array,
                    animal_dirs, labeled_array, label_map
                )

        plot_model_performance(
            pdf, data, uniref, img_array, animal_dirs, labeled_array, label_map
        )

if __name__ == '__main__':

    main()
