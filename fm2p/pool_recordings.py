import numpy as np
import pandas as pd
import fm2p
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_filler_series(nc, ni):
    # nc is number of number of cells
    # ni is number of entries of the object
    s = pd.Series(np.zeros(nc))
    for i in s.index.values:
        s.iloc[i] = (np.zeros(ni)*np.nan).astype(object)
    return s


rmse = lambda y, y_pred: ((sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)) ** 0.5)


def calc_heatmap(data, c):
    x_vals = data['theta_interp']
    y_vals = data['phi_interp']
    rates = data['norm_spikes'][c,:]
    n_x = 20
    n_y = 20

    x_bins = np.linspace(np.nanpercentile(x_vals, 5), np.nanpercentile(x_vals, 95), num=n_x+1)
    y_bins = np.linspace(np.nanpercentile(y_vals, 5), np.nanpercentile(y_vals, 95), num=n_y+1)

    # 2D histogram of firing rates
    heatmap = np.zeros((n_y, n_x))

    # Compute average firing rate in each bin
    for i in range(n_x):
        for j in range(n_y):
            in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i+1]) & \
                    (y_vals >= y_bins[j]) & (y_vals < y_bins[j+1])
            # if np.any(in_bin):
            heatmap[j, i] = np.nanmean(rates[in_bin])

    return heatmap, x_bins, y_bins


def pool_recordings():

    rec_list = [
        {
            'dir': r'K:\Mini2P\250616_DMM_DMM042_ltdk\fm2',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250618_DMM_DMM042_ltdk\fm2',
            'AP': -4,
            'ML': 4
        },
        {
            'dir': r'K:\Mini2P\250619_DMM_DMM042_ltdk\fm3',
            'AP': -4,
            'ML': 4
        },
        {
            'dir': r'K:\Mini2P\250626_DMM_DMM037_ltdk\fm1',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250626_DMM_DMM041_ltdk\fm1',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm3', 
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm5',
            'AP': -8,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250627_DMM_DMM041_ltdk\fm1',
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250627_DMM_DMM042_ltdk\fm2',
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250627_DMM_DMM042_ltdk\fm3',
            'AP': -8,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM037_ltdk\fm2',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM037_ltdk\fm3',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM041_ltdk\fm2',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM041_ltdk\fm4',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM042_ltdk\fm3',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'K:\Mini2P\250628_DMM_DMM042_ltdk\fm4',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250630_DMM_DMM037_ltdk\fm1',
            'AP': 0,
            'ML': -4
        },
        {
            'dir': r'K:\Mini2P\250630_DMM_DMM037_ltdk\fm3',
            'AP': +8,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250630_DMM_DMM041_ltdk\fm1',
            'AP': 8,
            'ML': 0
        },
        {
            'dir': r'K:\Mini2P\250630_DMM_DMM041_ltdk\fm3',
            'AP': 0,
            'ML': -4
        }
    ]
    
    merged_df = pd.DataFrame()

    for i, v in tqdm(enumerate(rec_list)):

        dir_ = v['dir']
        AP = v['AP']
        ML = v['ML']

        base_name = dir_.split('\\')[-2][:-5]
        rnum = dir_.split('\\')[-1][-1]
        revcorr_path = '{}_fm_{:02d}_revcorr_results_v4.h5'.format(base_name, int(rnum))
        if not os.path.isfile(os.path.join(dir_, revcorr_path)):
            revcorr_path = '{}_fm_{:02d}_revcorr_results_v3.h5'.format(base_name, int(rnum))
            if not os.path.isfile(os.path.join(dir_, revcorr_path)):
                revcorr_path = '{}_fm_{:02d}_revcorr_results.h5'.format(base_name, int(rnum))
        preproc_path = '{}_fm_{:02d}_preproc.h5'.format(base_name, int(rnum))
        animal = base_name.split('_')[2]
        date = base_name.split('_')[0]
        
        revcorr_data = fm2p.read_h5(os.path.join(dir_, revcorr_path))
        preproc_data = fm2p.read_h5(os.path.join(dir_, preproc_path))

        peth_dict = fm2p.calc_PETHs(preproc_data)

        df = pd.DataFrame()

        n_cells = np.size(preproc_data['norm_spikes'], 0)

        for c in range(n_cells):
            calc_heatmap(preproc_data, i)

        emptyseries = make_filler_series(n_cells, 12)

        for state in ['light', 'dark']:
            for key in ['distance_to_pillar', 'egocentric', 'phi', 'retinocentric', 'theta', 'yaw']:
                
                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    try:
                        s.iloc[c] = (revcorr_data[state][key]['tuning_curve'][c,:]).astype(object)
                    except KeyError:
                        try:
                            s.iloc[c] = (revcorr_data[state][key]['tunings'][c,:]).astype(object)
                        except KeyError:
                            s.iloc[c] = (emptyseries.copy() * np.nan).astype(object)
                df['{}_{}_tuning_curve'.format(key,state)] = s

                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    try:
                        s.iloc[c] = (revcorr_data[state][key]['tuning_stderr'][c,:]).astype(object)
                    except KeyError:
                        s.iloc[c] = (emptyseries.copy() * np.nan).astype(object)
                df['{}_{}_tuning_err'.format(key,state)] = s

                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    try:
                        s.iloc[c] = (revcorr_data[state][key]['tuning_bins']).astype(object)
                    except KeyError:
                        s.iloc[c] = (emptyseries.copy() * np.nan).astype(object)
                df['{}_{}_tuning_bins'.format(key,state)] = s

        peth_keys = [
            'right_PETHs',
            'left_PETHs',
            'up_PETHs',
            'down_PETHs',
            'norm_right_PETHs',
            'norm_left_PETHs',
            'norm_up_PETHs',
            'norm_down_PETHs'
        ]
        for k in peth_keys:
            s = make_filler_series(n_cells, 12)
            for c in range(n_cells):
                s.iloc[c] = (peth_dict[k][c,:])
            df[k[:-1]] = s

        df['xloc'] = np.zeros(n_cells)
        df['yloc'] = np.zeros(n_cells)
        for c in range(n_cells):
            df['xloc'].iloc[c] = np.median(preproc_data['cell_x_pix'][str(c)])
            df['yloc'].iloc[c] = np.median(preproc_data['cell_y_pix'][str(c)])

        df['cell_num'] = df.index.values
        df['ML_offset'] = - ML*100
        df['AP_offset'] = AP*100

        df['base_name'] = base_name
        df['recnum'] = rnum
        df['animal'] = animal
        df['date'] = date
        df['dir'] = dir_

        n_frames = np.size(preproc_data['norm_spikes'], 1)

        s_nspikes = make_filler_series(len(df), n_frames)
        s_rspikes = make_filler_series(len(df), n_frames)

        for c in range(n_cells):
            s_nspikes.iloc[c] = (preproc_data['norm_spikes'][c,:]).astype(object)
            s_rspikes.iloc[c] = (preproc_data['s2p_spks'][c,:]).astype(object)

        vars_to_add = [
            'ltdk_state_vec',
            'head_yaw_deg',
            'retinocentric',
            'egocentric',
            'head_x',
            'head_y',
            'longaxis',
            'norm_dFF',
            'theta_interp',
            'phi_interp',
            'raw_F',
            'twopT',
        ]
        for var in vars_to_add:
            s = make_filler_series(len(df), n_frames)
            for c in range(n_cells):
                s.iloc[0] = (preproc_data[var]).astype(object)
            df[var] = s

        vars_to_add = [
            'theta',
            'phi',
            'eyeT' # ,
            # 'rightward_onsets',
            # 'leftward_onsets',
            # 'upward_onsets',
            # 'downward_onsets'
        ]
        for var in vars_to_add:
            s = make_filler_series(len(df), len(preproc_data[var]))
            for c in range(n_cells):
                s.iloc[0] = (preproc_data[var]).astype(object)
            df[var] = s

        merged_df = pd.concat([merged_df, df], axis=0)

    merged_index = merged_df.reset_index()

    fm2p.write_group_h5(
        merged_index,
        r'K:\Mini2P\merged_V1PPC_dataset_w251010b.h5',
        repair_overflow=True
    )

    # merged_index = pd.read_hdf(r'K:\Mini2P\merged_V1PPC_dataset_251010.h5')

if __name__ == '__main__':
    pool_recordings()