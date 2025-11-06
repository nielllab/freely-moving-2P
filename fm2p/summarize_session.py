

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

plt.rcParams.update({'font.size': 6})


def summarize_session(preproc_path):

    data = fm2p.read_h5(preproc_path)

    usesuptitle = os.path.split(preproc_path)[1][:-11]

    pdf_savepath = os.path.join(
        os.path.split(preproc_path)[0],
        '{}_session_summary.pdf'.format(usesuptitle)
    )
    pdf = PdfPages(pdf_savepath)


    fig, axs = plt.subplots(5, 4, figsize=(8.5, 11.), dpi=300)

    axs[0,0].imshow(data['twop_mean_img'], cmap='gray')
    axs[0,0].set_title('{} cells'.format(np.size(data['norm_spikes'], 0)),fontsize=6)
    axs[0,0].axis('off')

    vars = ['head_yaw_deg', 'pitch_twop_interp','roll_twop_interp']
    is_running = np.append(np.array(data['speed']>2.), np.array(False))
    for i in range(3):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]
        axs[0,i+1].hist(data[var],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(data[var]) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        else:
            is_running1 = is_running
        axs[0,i+1].hist(data[var][is_running1],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            color='k', label='running'
        )
        axs[0,i+1].legend(frameon=False, fontsize=6)

    vars = ['gyro_z_twop_interp', 'gyro_x_twop_interp', 'gyro_y_twop_interp']
    for i in range(3):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]
        axs[1,i+1].hist(data[var],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(data[var]) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        else:
            is_running1 = is_running
        axs[1,i+1].hist(data[var][is_running1],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            color='k', label='running'
        )
        axs[1,i+1].legend(frameon=False, fontsize=6)

    axs[1,0].plot(data['eyeT'] - data['eyeT'][0], label='eyeT')
    axs[1,0].plot(data['imuT_trim'] - data['imuT_trim'][0], label='imuT')
    axs[1,0].plot(data['twopT'], label='twopT')
    axs[1,0].legend(frameon=False)
    axs[1,0].set_title('total time = {:.3} min'.format(np.max(data['twopT'])/60), fontsize=6)
    axs[1,0].set_xlabel('frames')
    axs[1,0].set_ylabel('time (s)')

    axs[4,0].hist(data['speed'], bins=np.linspace(0,100,40), color='k')
    axs[4,0].set_title('{:.4}% spent running'.format(np.sum((data['speed']>2)/ len(data['speed']))*100), fontsize=6)
    axs[4,0].set_xlim([0,100])
    axs[4,0].set_xlabel('speed (cm/s)', fontsize=6)

    axs[3,1].scatter(data['dHead'][::10], data['dEye'][::10], s=1, color='k')
    axs[3,1].set_xlim([-600,600])
    axs[3,1].set_ylim([-600,600])
    axs[3,1].set_xlim([-600,600])
    axs[3,1].set_ylim([-600,600])
    axs[3,1].plot([-600,600], [600,-600], color='tab:red', ls='--', alpha=0.3, lw=1)
    axs[3,1].set_xlabel('dHead (deg/s)', fontsize=6)
    axs[3,1].set_ylabel('dEye (deg/s)', fontsize=6)

    likelihoods = []
    for k in [x for x in data.keys() if '_likelihood' in x and 'bar' not in x and 'nose' not in x]:
        likelihoods.append(data[k])
    likelihoods = np.array(likelihoods)
    tracked_pts = np.sum(likelihoods > 0.6, 0)
    fracgoodframes = np.sum(tracked_pts >= 7) / np.size(tracked_pts)

    hist = axs[2,1].hist(tracked_pts+0.5, bins=8, color='k')
    axs[2,1].vlines(6.5, 0, np.max(hist[0]), color='tab:red', ls='--', lw=1)
    axs[2,1].set_title('{:.3}% tracked'.format(fracgoodframes*100))
    axs[2,1].set_xlabel('tracked pts')

    vars = ['theta', 'phi']
    for i in range(2):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]

        var_trim = np.rad2deg(data[var][data['eyeT_startInd']:data['eyeT_endInd']])
        eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
        eyeT = eyeT - eyeT[0]
        var_trim = fm2p.interpT(fm2p.interp_short_gaps(var_trim, max_gap=30), eyeT, data['twopT'])

        axs[2,i+2].hist(var_trim,
            bins=np.linspace(
                np.nanpercentile(var_trim, 5),
                np.nanpercentile(var_trim, 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(var_trim) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(var_trim) < len(is_running):
            is_running1 = is_running[:-1]
        else:
            is_running1 = is_running
        axs[2,i+2].hist(var_trim[is_running1],
            bins=np.linspace(
                np.nanpercentile(var_trim, 5),
                np.nanpercentile(var_trim, 95),
                30
            ),
            color='k', label='running'
        )
        axs[2,i+2].legend(frameon=False, fontsize=6)

        axs[4,i+2].plot((data['twopT']/60)[data['ltdk_state_vec']], var_trim[data['ltdk_state_vec']], color='tab:orange', lw=1, label='light')
        axs[4,i+2].plot((data['twopT']/60)[~data['ltdk_state_vec']], var_trim[~data['ltdk_state_vec']], color='navy', lw=1, label='dark')
        axs[4,i+2].legend(frameon=False)
        axs[4,i+2].set_ylabel('{} (deg)'.format(var))
        axs[4,i+2].set_xlabel('time (min)')

    axs[3,0].plot(data['head_x'], data['head_y'], 'k', lw=1)
    axs[3,0].set_xlabel('x (pxls)')
    axs[3,0].set_ylabel('y (pxls)')

    for i in range(len(data['axes_rel_cent_x']['0'][0]))[::2][:-1]:
        axs[2,0].plot(
            (
                data['axes_rel_cent_x']['0'][0][i],
                data['axes_rel_cent_x']['0'][1][i]
            ),

            (
                data['axes_rel_cent_y']['0'][0][i],
                data['axes_rel_cent_y']['0'][1][i]
            ),
            lw=1
        )
    axs[2,0].plot(data['camcent'][0], data['camcent'][1], '*', color='tab:red', label='camera center')
    axs[2,0].legend(frameon=False)
    axs[2,0].set_xlabel('theta (deg)')
    axs[2,0].set_ylabel('phi (deg)')

    axs[4,1].bar(0, len(data['comp_left']), color='tab:blue', label='left')
    axs[4,1].bar(0.8, len(data['comp_right']), color='tab:red', label='right')
    axs[4,1].bar(2, len(data['gaze_left']), color='tab:blue')
    axs[4,1].bar(2.8, len(data['gaze_right']), color='tab:red')
    axs[4,1].set_xticks([0.4, 2.4], labels=['compensatory','gaze shift'])
    axs[4,1].legend(frameon=False)
    axs[4,1].set_ylabel('# movements')

    axs[0,1].set_xlabel('yaw (deg)')
    axs[0,2].set_xlabel('pitch (deg)')
    axs[0,3].set_xlabel('roll (deg)')

    axs[1,1].set_xlabel('dYaw (deg/s)')
    axs[1,2].set_xlabel('dPitch (deg/s)')
    axs[1,3].set_xlabel('dRoll (deg/s)')

    axs[2,2].set_xlabel('theta (deg)')
    axs[2,3].set_xlabel('phi (deg)')

    axs[3,2].set_xlabel('dEye (deg/s)')
    axs[3,3].set_xlabel('dGaze (deg/s)')

    vars = ['dEye', 'dGaze']
    for i in range(2):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]

        eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
        eyeT = eyeT - eyeT[0]
        t = eyeT.copy()[:-1]
        t1 = t + (np.diff(eyeT) / 2)
        var_trim = fm2p.interpT(fm2p.interp_short_gaps(data[var], max_gap=30), t1, data['twopT'])

        axs[3,i+2].hist(var_trim,
            bins=np.linspace(
                np.nanpercentile(var_trim, 5),
                np.nanpercentile(var_trim, 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(var_trim) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(var_trim) < len(is_running):
            is_running1 = is_running[:-1]
        else:
            is_running1 = is_running
        axs[3,i+2].hist(var_trim[is_running1],
            bins=np.linspace(
                np.nanpercentile(var_trim, 5),
                np.nanpercentile(var_trim, 95),
                30
            ),
            color='k', label='running'
        )
        axs[3,i+2].legend(frameon=False, fontsize=6)

    fig.suptitle(usesuptitle)

    fig.tight_layout()

    pdf.savefig(fig)
    pdf.close()

    print('Summary PDF saved to {}'.format(pdf_savepath))



if __name__ == '__main__':


    preproc_path = fm2p.select_file(
        'Select preprocessing HDF file.',
        filetypes=[('HDF','.h5'),])

    summarize_session(preproc_path)

    