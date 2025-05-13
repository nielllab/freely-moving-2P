
import numpy as np
from scipy import io
import itertools

import fm2p
import imgtools


def get_independent_axons(matpath, cc_thresh=0.5, apply_dFF_filter=False):

    mat = io.loadmat(matpath)
    dFF = mat['data'].item()[-1].copy()

    if apply_dFF_filter:
        # Smooth dFF traces of all cells
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = imgtools.nanmedfilt(
                    imgtools.rolling_average_1d(dFF[c,:], 11),
            25).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)

    # Calculate all between-cell correlation coeffients
    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat,0)])
    if apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = fm2p.corr2_coeff(
                all_smoothed_units[perm_mat[i,0]][np.newaxis,:],
                all_smoothed_units[perm_mat[i,1]][np.newaxis,:]
            )
    elif not apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = fm2p.corr2_coeff(
                dFF[perm_mat[i,0]][np.newaxis,:],
                dFF[perm_mat[i,1]][np.newaxis,:]
            )

    # Find axon pairs with cc above threshold
    check_index = np.where(cc_vec > cc_thresh)[0]
    exclude_inds = []

    for c in check_index:
        axon1 = perm_mat[c,0]
        axon2 = perm_mat[c,1]
        # exclude the neuron with the lower integrated dFF
        if (np.sum(dFF[axon1,:]) < np.sum(dFF[axon2,:])):
            exclude_inds.append(axon1)
        elif (np.sum(dFF[axon1,:]) > np.sum(dFF[axon2,:])):
            exclude_inds.append(axon2)

    exclude_inds = list(set(exclude_inds))
    usecells = [c for c in list(np.arange(np.size(dFF,0))) if c not in exclude_inds]

    # Check correlation between global frame fluorescence and the dF/F of each axon
    frameF = mat['data'].item()[7].copy()
    gcc_vec = np.zeros([len(usecells)])
    for i,c in enumerate(usecells):
        gcc_vec[i] = fm2p.corr2_coeff(
            dFF[c,:][np.newaxis,:],
            frameF
        )

    axon_correlates_with_globalF = np.where(gcc_vec > 0)[0]
    usecells_gcc = [c for c in usecells if c in axon_correlates_with_globalF]

    print(usecells)
    print(usecells_gcc)

    dFF_out = dFF.copy()[usecells_gcc, :]

    return dFF_out
    
