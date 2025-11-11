
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import correlate
import gc

import fm2p
import imgtools


def find_delay_frames(stim_s, pop_s, max_lag=80):

    stim_s = np.asarray(stim_s).ravel()
    pop_s  = np.asarray(pop_s).ravel()

    stim_s = (stim_s - np.mean(stim_s)) / np.std(stim_s)
    pop_s = (pop_s - np.mean(pop_s)) / np.std(pop_s)

    corr = correlate(stim_s, pop_s, mode='full')
    lags = np.arange(-len(stim_s)+1, len(pop_s))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = lags[mask][np.argmax(corr[mask])]

    return lag


def compute_calcium_sta_spatial(
    stimulus,
    spikes,
    stim_times,
    spike_times,
    window=20,
    auto_delay=True,
    max_lag_frames=80,
):
    
    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)
    
    # trim off extra frames at end of 2P data
    stimend = np.size(stimulus,0)/2
    spikeend, _ = fm2p.find_closest_timestamp(spike_times, stimend)
    spikes = spikes[:,:spikeend]
    spike_times = spike_times[:spikeend]

    nFrames, stimY, stimX = np.shape(stimulus)

    stim_mean_trace = np.mean(stimulus, axis=(1,2))

    bg_est = np.median(stimulus)
    white_mask = (stimulus > bg_est)
    black_mask = (stimulus < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    flat_signed = np.reshape(signed_stim, [nFrames, stimY*stimX])
    flat_signed = flat_signed - np.mean(flat_signed, axis=0, keepdims=True)

    n_stim, n_features = flat_signed.shape
    n_cells, n_spike_samples = spikes.shape

    if n_spike_samples != len(spike_times):
        raise ValueError(f"spikes.shape[1] ({n_spike_samples}) != len(spike_times) ({len(spike_times)})")

    pop_trace = np.mean(spikes, axis=0)

    bin_edges = np.concatenate([
        stim_times,
        [stim_times[-1] + np.median(np.diff(stim_times))],
    ])
    pop_sum, _ = np.histogram(spike_times, bins=bin_edges, weights=pop_trace)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    counts[counts == 0] = 1
    pop_rate_per_frame = pop_sum / counts

    est_delay_frames = 0
    if auto_delay:
        est_delay_frames = find_delay_frames(
            stim_mean_trace,
            pop_rate_per_frame,
            max_lag=max_lag_frames
        )
        delay = est_delay_frames * np.median(np.diff(stim_times))

        stim_times_shifted = stim_times + (delay if delay is not None else 0.0)
    else:
        stim_times_shifted = stim_times

    sta_all = np.zeros((n_cells, window + 1, n_features))
    eps = 1e-9

    for cell_idx in tqdm(range(n_cells)):
        cell_spikes = spikes[cell_idx,:]

        interp_fn = interp1d(
            spike_times,
            cell_spikes,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True
        )
        spike_rate_per_frame = interp_fn(stim_times_shifted)

        sta = np.zeros((window + 1, n_features))
        total_rate = 0.

        for i, rate in enumerate(spike_rate_per_frame):
            if rate <= 0 or i < window or i + window + 1 >= n_stim:
                continue

            stim_segment = flat_signed[i - window : i+1 ]
 
            sta += rate * stim_segment
            total_rate += rate
                
        sta /= (total_rate + eps)
        sta_all[cell_idx] = sta

    lag_axis = np.arange(-window, window + 1)

    return sta_all, lag_axis, est_delay_frames


def calc_sparse_noise_STAs(preproc_path, stimpath=None):

    if stimpath is None:
        stimpath = r'T:\dylan\sparse_noise_sequence_v7.npy'
    stimulus = np.load(stimpath)[:,:,:,0]

    data = fm2p.read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    # calculate the STAs
    sta_all, lag_axis, delay = compute_calcium_sta_spatial(
        stimulus,
        norm_spikes,
        stimT,
        twopT,
        window=15,
        auto_delay=False
    )

    n_cells = np.size(norm_spikes, 0)

    # find best lag so there is only one STA per cell to deal with
    best_sta = np.zeros([
        n_cells,
        np.size(sta,2),
        np.size(sta,3)
    ])
    best_lags = np.zeros(n_cells)

    for c in range(n_cells):
        lagmax = np.zeros(16) * np.nan
        for l in range(16):
            lagmax[l] = np.nanmax(sta[c,l,:,:])
        best_lags[c] = np.nanargmax(lagmax)

    del sta
    gc.collect()

    n_chunks = 20
    split_frac = 0.5

    n_samps = np.size(stimulus, 0)
    chunk_size = n_samps // n_chunks
    all_inds = np.arange(0, n_samps)
    chunk_order = np.arange(n_chunks)
    np.random.shuffle(chunk_order)
    split_bound = int(n_chunks * split_frac)

    inds_a = []
    inds_b = []
    for cnk_i, cnk in enumerate(chunk_order):
        _inds = all_inds[(chunk_size*cnk) : ((chunk_size*(cnk+1)))]
        if cnk_i < split_bound:
            inds_a.extend(_inds)
        elif cnk_i >= split_bound:
            inds_b.extend(_inds)
    inds_a = np.sort(np.array(inds_a)).astype(int)
    inds_b = np.sort(np.array(inds_b)).astype(int)

    stim_a = stimulus[inds_a,:,:]
    stim_b = stimulus[inds_b,:,:]

    spikes_a = norm_spikes[:,inds_a]
    spikes_b = norm_spikes[:,inds_b]

    spikeT_a = 

    stimT_a = 

    

    compute_calcium_sta_spatial(
        stimulus=
        spikes=
        stim_times=
        spike_times=
    )

    

