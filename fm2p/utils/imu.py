
import numpy as np
import pandas as pd
import fm2p


def read_IMU(vals_path, time_path):

    imuT = fm2p.read_timestamp_file(time_path)

    df = pd.read_csv(vals_path, header=None)
    df.columns = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']


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

    # imu_dict = {
    #     'acc_x': df['ax'].to_numpy(),
    #     'acc_y': df['ay'].to_numpy(),
    #     'acc_z': df['az'].to_numpy(),
    #     'gyro_x': df['gx'].to_numpy(),
    #     'gyro_y': df['gy'].to_numpy(),
    #     'gyro_z': df['gz'].to_numpy(),
    #     'pitch': df['pitch'].to_numpy(),
    #     'roll': df['roll'].to_numpy(),
    #     'imuT': imuT
    # }

    return df, imuT