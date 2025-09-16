
import numpy as np
import pandas as pd
import fm2p


def read_IMU(vals_path, time_path):

    imuT = fm2p.read_timestamp_file(time_path)

    df = pd.read_csv(vals_path, header=None)
    df.columns = [
        'acc_y',
        'acc_z',
        'acc_x',
        'gyro_y',
        'gyro_z',
        'gyro_x'
    ]

    # Invert sign of channels
    df['gyro_y'] = -df['gyro_y']
    df['gyro_z'] = -df['gyro_z']

    n_samps = np.size(df,0)
    roll_pitch = np.zeros([n_samps, 2])

    IMU = fm2p.ImuOrientation()

    for i in range(n_samps):
        roll_pitch[i] = IMU.process((
            df[['acc_x','acc_y','acc_z']].iloc[i].to_numpy(),
            df[['gyro_x','gyro_y','gyro_z']].iloc[i].to_numpy(),
        ))

    df['roll'] = roll_pitch[:,0]
    df['pitch'] = roll_pitch[:,1]

    return df, imuT