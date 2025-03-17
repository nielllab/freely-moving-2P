

import os
import numpy as np
import argparse

import fm2p

def summarize_model_fit():

    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default=None)
    parser.add_argument('--preproc', type=str, default=None)
    parser.add_argument('--nulldir', type=str, default=None)
    args = parser.parse_args()

    if args.modeldir is None:
        print('Choose model fit directory (subdirectory within a recording directory).')
        model_dir = fm2p.select_directory(
            title='Choose a model fit directory.'
        )
    else:
        model_dir = args.modeldir

    if args.preproc is None:
        print('Choose a preprocessing file.')
        preproc_path = fm2p.select_file(
            title='Choose a preprocessing file.',
            filetypes=[('H5','.h5')]
        )
    else:
        preproc_path = args.preproc

    if args.nulldir is None:
        print('Select the null model fit directory.')
        null_dir = fm2p.select_directory(
            title='Select the null model fit directory.'
        )
    else:
        null_dir = args.nulldir


    print('Reading in model fit results.')
    # model3_path = os.path.join(models_dir, 'LNLP_fit_results_multihot')
    model = fm2p.read_models(model_dir)
    savepath = os.path.join(model_dir, 'cell_summary_LNP_v01.pdf')

    ego_bins = np.deg2rad(np.linspace(-180, 180, 36))
    retino_bins = np.deg2rad(np.linspace(-180, 180, 36))
    pupil_bins = np.deg2rad(np.linspace(0, 100, 10))

    var_bins = [pupil_bins, retino_bins, ego_bins]

    print('Reading null model fit results.')
    model_null = fm2p.read_models(null_dir)

    print('Reading in preprocessed experiment data.')
    preprocdata = fm2p.read_h5(preproc_path)

    print('Writing summary file.')
    fm2p.write_detailed_cell_summary(
        model,
        var_bins=var_bins,
        savepath=savepath,
        preprocdata=preprocdata,
        null_data=model_null,
        lag_val=0
    )

if __name__ == '__main__':

    summarize_model_fit()
