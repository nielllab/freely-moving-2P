
import os
import cv2
import json
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys as fme


class Eyecam(fme.Camera):
    """ Preprocessing for head-mounted eye camera.

    Parameters
    ----------
    cfg : dict
        Dictionary of configuration parameters.
    recording_name : str
        Name of the recording.
    recording_path : str
        Path to the recording.
    camname : str
        Name of the camera.
    
    Methods
    -------
    fit_ellipse
        Fit an ellipse to points labeled around the perimeter of pupil.
    track_pupil
        Track the pupil in all video frame.
    eye_diagnostic_video
        Create a diagnostic video of the pupil tracking overlaying points
        and the ellipse over the video of behavior.
    sigmoid_curve
        Sigmoid curve function.
    sigmoid_fit
        Fit a sigmoid curve to data.
    get_torsion_from_ridges
        Get torsion (omega) from rotation of ridges along the edge of the pupil.
    save_params
        Save the NC file of parameters.
    process
        Run eyecam preprocessing.
        
    """


    def __init__(self, cfg, recording_name, recording_path, camname):
        fme.Camera.__init__(self, cfg, recording_name, recording_path, camname)

        self.eye_fig_pts_dwnspl = 100


    def fit_ellipse(self, x, y):
        """ Fit an ellipse to points labeled around the perimeter of pupil.

        Parameters
        ----------
        x : np.array
            Positions of points along the x-axis for a single video frame.
        y : np.array
            Positions of labeled points along the y-axis for a single video frame.

        Returns
        -------
        ellipse_dict : dict
            Parameters of the ellipse...
            X0 : center at the x-axis of the non-tilt ellipse
            Y0 : center at the y-axis of the non-tilt ellipse
            a : radius of the x-axis of the non-tilt ellipse
            b : radius of the y-axis of the non-tilt ellipse
            long_axis : radius of the long axis of the ellipse
            short_axis : radius of the short axis of the ellipse
            angle_to_x : angle from long axis to horizontal plane
            angle_from_x : angle from horizontal plane to long axis
            X0_in : center at the x-axis of the tilted ellipse
            Y0_in : center at the y-axis of the tilted ellipse
            phi : tilt orientation of the ellipse in radians

        """

        # Remove bias of the ellipse
        meanX = np.mean(x)
        meanY = np.mean(y)
        x = x - meanX
        y = y - meanY

        # Estimation of the conic equation
        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        a = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        # Extract parameters from the conic equation
        a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]

        # Eigen decomp
        Q = np.array([[a, b/2],[b/2, c]])
        eig_val, eig_vec = np.linalg.eig(Q)

        # Get angle to long axis
        if eig_val[0] < eig_val[1]:
            angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        else:
            angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])

        angle_from_x = angle_to_x
        orientation_rad = 0.5 * np.arctan2(b, (c-a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

        a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                        0,
                        a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                        d*cos_phi - e*sin_phi,
                        d*sin_phi + e*cos_phi]

        meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                        sin_phi*meanX + cos_phi*meanY]

        # Check if conc expression represents an ellipse
        test = a*c

        if test > 0:

            # Make sure coefficients are positive
            if a<0:
                a, c, d, e = [-a, -c, -d, -e]

            # Final ellipse parameters
            X0 = meanX - d/2/a
            Y0 = meanY - e/2/c
            F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
            a = np.sqrt(F/a)
            b = np.sqrt(F/c)
            long_axis = 2*np.maximum(a,b)
            short_axis = 2*np.minimum(a,b)

            # Rotate axes backwards to find center point of
            # original tilted ellipse
            R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
            P_in = R @ np.array([[X0],[Y0]])
            X0_in = P_in[0][0]
            Y0_in = P_in[1][0]

            # Organize parameters in dictionary to return
            ellipse_dict = {
                'X0':X0,
                'Y0':Y0,
                'F':F,
                'a':a,
                'b':b,
                'long_axis':long_axis/2,
                'short_axis':short_axis/2,
                'angle_to_x':angle_to_x,
                'angle_from_x':angle_from_x,
                'cos_phi':cos_phi,
                'sin_phi':sin_phi,
                'X0_in':X0_in,
                'Y0_in':Y0_in,
                'phi':orientation_rad
            }

        else:

            # If the conic equation didn't return an ellipse, do not
            # return any real values and fill the dictionary with NaNs.
            dict_keys = ['X0','Y0','F','a','b','long_axis',
                         'short_axis','angle_to_x','angle_from_x',
                         'cos_phi','sin_phi','X0_in','Y0_in','phi']
            dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

            ellipse_dict = dict(zip(dict_keys, dict_vals))
        
        return ellipse_dict


    def track_pupil(self):
        """ Track the pupil in the current recording.
        """

        # Set up the pdf to be saved out with diagnostic figures
        pdf_name = '{}_{}_tracking_figs.pdf'.format(self.recording_name, self.camname)
        pdf = PdfPages(os.path.join(self.recording_path, pdf_name))
        
        # If this is a head-fixed recording, read in existing freely moving camera
        # center, scale, etc. The pipeline will process all fm recordings first,
        # so it will be possible to read in fm camera calibration parameters for
        # every hf recording.
        if self.cfg['share_eyecal'] is True:

            if 'hf' in self.recording_name:

                # (Should always go for fm1 before fm2)
                path_to_existing_props = sorted(fme.find('*fm_eyecameracalc_props.json',
                                                             self.cfg['animal_directory']))
                
                if len(path_to_existing_props) == 0:
                    print('Found no eyecam calibration json')
                    path_to_existing_props = None

                elif len(path_to_existing_props) == 1:
                    print('Found no eyecam calibration json')
                    path_to_existing_props = path_to_existing_props[0]

                elif len(path_to_existing_props) > 1:
                    print('Found multiple eyecam calibration json')
                    print(path_to_existing_props[0])
                    path_to_existing_props = path_to_existing_props[0]

                if path_to_existing_props is not None:
                    with open(path_to_existing_props, 'r') as fp:
                        existing_camera_calib_props = json.load(fp)

                elif path_to_existing_props is None:

                    # If a json of paramters can't be found, though, we'll
                    # get these values for the hf recording.
                    existing_camera_calib_props = None

            elif 'fm' in self.recording_name or self.cfg['strict_dir']:
                existing_camera_calib_props = None
            
            else:
                existing_camera_calib_props = None

        elif self.cfg['share_eyecal'] is False:
            
            calibration_param_file = next(i for i in fme.find('*fm_eyecameracalc_props.json',  \
                                                        self.cfg['recording_path']) if self.camname in i)
            
            with open(calibration_param_file, 'r') as fp:
                existing_camera_calib_props = json.load(fp)

        # Names of the different points
        x_vals, y_vals, likeli_vals = self.split_xyl()
        likelihood_in = likeli_vals.values

        
        if self.cfg['eye_lght'] and self.cfg['eye_lghtsub']:

            # Subtract center of IR light reflection from points around the pupil.

            spot_xcent = np.mean(x_vals.iloc[:,-5:], 1)
            spot_ycent = np.mean(y_vals.iloc[:,-5:], 1)
            spot_likelihood = likelihood_in[:,-5:].copy()

            likelihood = likelihood_in[:,:-5]

            x_vals = x_vals.iloc[:,:-5].subtract(spot_xcent, axis=0)
            y_vals = y_vals.iloc[:,:-5].subtract(spot_ycent, axis=0)

        elif self.cfg['eye_lght'] and not self.cfg['eye_lghtsub']:

            # Drop the IR light pts without subtracting
            x_vals = x_vals.iloc[:,:-5]
            y_vals = y_vals.iloc[:,:-5]
            likelihood = likelihood_in[:,:-5]

        elif self.cfg['eye_crnrs_1st']:

            # Handle points if they were labaled in an unexpected order.
            x_vals = x_vals.iloc[:,2:]
            y_vals = y_vals.iloc[:,2:]
            likelihood = likelihood_in[:,2:]
        
        # Drop tear/outer eye points
        if self.cfg['eye_crnr']:

            if self.cfg['eye_lght'] and self.cfg['eye_lghtsub']:
                x_vals = x_vals.iloc[:,:-2]
                y_vals = y_vals.iloc[:,:-2]
                likelihood = likelihood[:,:-2]

            if not self.cfg['eye_lght'] and self.cfg['eye_lghtsub']:
                x_vals = x_vals.iloc[:,:-2]
                y_vals = y_vals.iloc[:,:-2]
                likelihood = likelihood_in[:,:-2]

        else:
            likelihood = likelihood_in

        pupil_count = np.sum(likelihood >= self.cfg['Lthresh'], 1)

        # Get bools of when a frame is usable with the right number of
        # points above liklelihood threshold
        if self.cfg['eye_lghtsub']:
            
            # If spot subtraction is being done, we should only include frames
            # where all five pts marked around the ir spot are good (centroid
            # would be off otherwise)
            spot_count = np.sum(spot_likelihood >= self.cfg['Lthresh'], 1)

            usegood_eye = (pupil_count >= self.cfg['eye_useN']) &               \
                          (spot_count >= self.cfg['eye_lghtN'])

            usegood_eyecalib = (pupil_count >= self.cfg['eye_calN']) &          \
                               (spot_count >= self.cfg['eye_lghtN'])
            
            usegood_reflec = (spot_count >= self.cfg['eye_lghtN'])

        else:
            usegood_eye = pupil_count >= self.cfg['eye_useN']
            usegood_eyecalib = pupil_count >= self.cfg['eye_calN']
        
        # How well did reflection track?
        if self.cfg['eye_lghtsub']:
            plt.figure()
            plt.plot(spot_count[0:-1:10])
            plt.title('{:.3}% good'.format(np.mean(usegood_reflec)*100))
            plt.ylabel('num good reflection points')
            plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()

        # How well did eye track?
        plt.figure()
        plt.plot(pupil_count[0:-1:10])
        plt.title('{:.3}% good'.format(np.mean(usegood_eye)*100))
        plt.ylabel('num good pupil points')
        plt.xlabel('every 10th frame')
        pdf.savefig()
        plt.close()

        # Hist of eye tracking quality
        plt.figure()
        plt.hist(pupil_count, bins=9,
                        range=(0,9), density=True)
        plt.xlabel('num good eye points')
        plt.ylabel('fraction of frames')
        pdf.savefig()
        plt.close()

        # Threshold out pts more than a given distance away from nanmean of that point
        std_thresh_x = np.empty(np.shape(x_vals))

        for point_loc in range(0,np.size(x_vals, 1)):
            _val = x_vals.iloc[:,point_loc]
            std_thresh_x[:,point_loc] = (np.abs(np.nanmean(_val) - _val)                \
                            / self.cfg['eye_pxl2cm']) > self.cfg['eye_distthresh']

        std_thresh_y = np.empty(np.shape(y_vals))

        for point_loc in range(0,np.size(x_vals, 1)):
            _val = y_vals.iloc[:,point_loc]
            std_thresh_y[:,point_loc] = (np.abs(np.nanmean(_val) - _val)                \
                            / self.cfg['eye_pxl2cm']) > self.cfg['eye_distthresh']

        std_thresh_x = np.nanmean(std_thresh_x, 1)
        std_thresh_y = np.nanmean(std_thresh_y, 1)

        x_vals[std_thresh_x > 0] = np.nan
        y_vals[std_thresh_y > 0] = np.nan

        cols = [
            'X0',           # 0
            'Y0',           # 1
            'F',            # 2
            'a',            # 3
            'b',            # 4
            'long_axis',    # 5
            'short_axis',   # 6
            'angle_to_x',   # 7
            'angle_from_x', # 8
            'cos_phi',      # 9
            'sin_phi',      # 10
            'X0_in',        # 11
            'Y0_in',        # 12
            'phi'           # 13
        ]

        ellipse = np.empty([len(usegood_eye), 14])
        
        # Step through each frame, fit an ellipse to points, and add ellipse
        # parameters to array with data for all frames together.
        linalgerror = 0
        for step in tqdm(range(0,len(usegood_eye))):
            
            if usegood_eye[step] == True:
                
                try:

                    e_t = self.fit_ellipse(x_vals.iloc[step].values,
                                           y_vals.iloc[step].values)
                    
                    ellipse[step] = [
                        e_t['X0'],              # 0
                        e_t['Y0'],              # 1
                        e_t['F'],               # 2
                        e_t['a'],               # 3
                        e_t['b'],               # 4
                        e_t['long_axis'],       # 5
                        e_t['short_axis'],      # 6
                        e_t['angle_to_x'],      # 7
                        e_t['angle_from_x'],    # 8
                        e_t['cos_phi'],         # 9
                        e_t['sin_phi'],         # 10
                        e_t['X0_in'],           # 11
                        e_t['Y0_in'],           # 12
                        e_t['phi']              # 13
                    ]
                
                except np.linalg.LinAlgError as e:

                    linalgerror = linalgerror + 1
                    ellipse[step] = list(np.ones([len(cols)]) * np.nan)
            
            elif usegood_eye[step] == False:

                ellipse[step] = list(np.ones([len(cols)]) * np.nan)

        print('LinAlg error count = ' + str(linalgerror))
        
        # List of all places where the ellipse meets threshold
        R = np.linspace(0, 2*np.pi, 100)

        # (short axis / long axis) < thresh
        usegood_ellipcalb = np.where((usegood_eyecalib == True)                     \
                & ((ellipse[:,6] / ellipse[:,5]) < self.cfg['eye_ellthresh']))
        
        # Limit number of frames used for calibration
        f_lim = 50000
        if np.size(usegood_ellipcalb,1) > f_lim:
            shortlist = sorted(np.random.choice(usegood_ellipcalb[0],
                                size=f_lim, replace=False))
        else:
            shortlist = usegood_ellipcalb
        
        # Find camera center
        A = np.vstack([np.cos(ellipse[shortlist,7]),
                       np.sin(ellipse[shortlist,7])])

        b = np.expand_dims(np.diag(A.T @ np.squeeze(ellipse[shortlist, 11:13].T)), axis=1)
        
        # Only use the camera center from this recording if values were not
        # read in from a json. In practice, this means hf recordings have
        # their cam center thrown out and use the fm values read in.
        if existing_camera_calib_props is None:
            cam_cent = np.linalg.inv(A @ A.T) @ A @ b

        elif existing_camera_calib_props is not None:
            cam_cent = np.array([
                [float(existing_camera_calib_props['cam_cent_x'])],
                [float(existing_camera_calib_props['cam_cent_y'])]
            ])
        
        # Ellipticity and scale
        ellipticity = (ellipse[shortlist,6] / ellipse[shortlist,5]).T
        
        if existing_camera_calib_props is None:
            
            try:
                scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
                (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=0)))       \
                / np.sum(1 - (ellipticity)**2)
            
            except ValueError:

                scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
                (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=1)))       \
                / np.sum(1 - (ellipticity)**2)

        elif existing_camera_calib_props is not None:

            scale = float(existing_camera_calib_props['scale'])
        
        # Pupil angles

        # Horizontal orientation (THETA)
        theta = np.arcsin((ellipse[:,11] - cam_cent[0]) / scale)

        # Vertical orientation (PHI)
        phi = np.arcsin((ellipse[:,12] - cam_cent[1]) / np.cos(theta) / scale)
        

        try:
            plt.figure()
            plt.plot(np.rad2deg(phi)[0:-1:10])
            plt.title('phi')
            plt.ylabel('deg')
            plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.plot(np.rad2deg(theta)[0:-1:10])
            plt.title('theta')
            plt.ylabel('deg')
            plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()

        except Exception as e:
            print('Figure error for plots of theta, phi')
            print(e)

        # Organize data to return as an xarray of most essential parameters
        ellipse_df = pd.DataFrame({
            'theta':list(theta),
            'phi':list(phi),
            'longaxis':list(ellipse[:,5]),
            'shortaxis':list(ellipse[:,6]),
            'X0':list(ellipse[:,11]),
            'Y0':list(ellipse[:,12]),
            'ellipse_phi':list(ellipse[:,7])
        })

        ellipse_param_names = [
            'theta',
            'phi',
            'longaxis',
            'shortaxis',
            'X0',
            'Y0',
            'ellipse_phi'
        ]

        ellipse_out = xr.DataArray(ellipse_df,
            coords=[('frame', range(0, len(ellipse_df))),
                    ('ellipse_params', ellipse_param_names)],
            dims=['frame', 'ellipse_params'])

        ellipse_out.attrs['cam_center_x'] = cam_cent[0,0]
        ellipse_out.attrs['cam_center_y'] = cam_cent[1,0]
        

        # Ellipticity histogram
        fig_dwnsmpl = 100

        try:
            # Hist of ellipticity
            plt.figure()
            plt.hist(ellipticity, density=True)
            plt.title('ellipticity; thresh='+str(self.cfg['eye_ellthresh']))
            plt.ylabel('ellipticity')
            plt.xlabel('fraction of frames')
            pdf.savefig()
            plt.close()
            
            # Eye axes relative to center
            w = ellipse[:,7]
            plt.figure()
            for i in range(0,len(usegood_ellipcalb)):

                _show = usegood_ellipcalb[i::fig_dwnsmpl]

                plt.plot((ellipse[_show,11] + [-5 * np.cos(w[_show]),       \
                         5 * np.cos(w[_show])]),                            \
                         (ellipse[_show,12] + [-5*np.sin(w[_show]),         \
                         5*np.sin(w[_show])]))

            plt.plot(cam_cent[0], cam_cent[1], 'r*')
            plt.title('eye axes relative to center')
            pdf.savefig()
            plt.close()

        except Exception as e:
            print('Figure error in plots of ellipticity and axes relative to center')
            print(e)
            
        # Check calibration
        try:

            xvals = np.linalg.norm(ellipse[usegood_eyecalib, 11:13].T - cam_cent, axis=0)

            yvals = scale * np.sqrt( 1 - (ellipse[usegood_eyecalib, 6]              \
                                        / ellipse[usegood_eyecalib, 5]) **2)

            calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)

            slope, _, r_value, _, _ = scipy.stats.linregress(xvals[calib_mask],
                                                             yvals[calib_mask].T)
        
        except ValueError:
            print('no good frames that meet criteria... check DLC tracking!')

        # Save out camera center and scale as np array (but only if this is
        # a freely moving recording).
        if 'fm' in self.recording_name or not self.cfg['strict_dir']:
            
            calib_props_dict = {
                'cam_cent_x':float(cam_cent[0]),
                'cam_cent_y':float(cam_cent[1]),
                'scale':float(scale),
                'regression_r':float(r_value),
                'regression_m':float(slope)
            }

            _savename = '{}{}_fm_eyecameracalc_props.json'.format(self.recording_name,
                                                                  self.camname)

            calib_props_dict_savepath = os.path.join(self.recording_path, _savename)

            print('Saving calibration parameters to '+ calib_props_dict_savepath)
            
            with open(calib_props_dict_savepath, 'w') as f:
                json.dump(calib_props_dict, f)
        
        # Figures of scale and center
        try:
            plt.figure()
            plt.plot(xvals[::fig_dwnsmpl],
                     yvals[::fig_dwnsmpl], '.', markersize=1)
            plt.plot(np.linspace(0,50), np.linspace(0,50), 'r')
            plt.title('scale={:.3} r={:.3} m={:.3}'.format(scale, r_value, slope))
            plt.xlabel('pupil camera dist')
            plt.ylabel('scale * ellipticity')
            pdf.savefig()
            plt.close()

            # Calibration of camera center
            delta = (cam_cent - ellipse[:, 11:13].T)

            _useec = usegood_eyecalib[::fig_dwnsmpl]
            _use3 = np.squeeze(usegood_ellipcalb)[::fig_dwnsmpl]

            plt.figure()
            plt.plot(np.linalg.norm(delta[:,_useec], 2, axis=0),                \
                    ((delta[0, _useec].T * np.cos(ellipse[_useec, 7]))          \
                    + (delta[1, _useec].T * np.sin(ellipse[_useec, 7])))        \
                    / np.linalg.norm(delta[:, _useec], 2, axis=0).T,            \
                    'y.', markersize=1)

            plt.plot(np.linalg.norm(delta[:,_use3], 2, axis=0),                 \
                    ((delta[0, _use3].T * np.cos(ellipse[_use3,7]))             \
                    + (delta[1, _use3].T * np.sin(ellipse[_use3, 7])))          \
                    / np.linalg.norm(delta[:, _use3], 2, axis=0).T,             \
                    'r.', markersize=1)

            plt.title('camera center calibration')
            plt.ylabel('abs([PC-EC]).[cosw;sinw]')
            plt.xlabel('abs(PC-EC)')

            patch0 = mpatches.Patch(color='y', label='all pts')
            patch1 = mpatches.Patch(color='y', label='calibration pts')
            plt.legend(handles=[patch0, patch1])

            pdf.savefig()
            plt.close()

        except Exception as e:
            print('Figure error in plots of scale and camera center')
            print(e)

        pdf.close()

        self.ellipse_params = ellipse_out


    def eye_diagnostic_video(self):
        """ Plot video of eye tracking.
        """

        # Read in video, set up save file
        vidread = cv2.VideoCapture(self.video_path)
        width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

        _vidname = '{}_{}_plot.avi'.format(self.recording_name, self.camname)
        savepath = os.path.join(self.recording_path, _vidname)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

        # Only do the first number of frames (limit of frames to use should
        # be set in cfg dict)
        nFrames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.cfg['save_frameN'] > nFrames:
            num_save_frames = nFrames
        else:
            num_save_frames = self.cfg['save_frameN']

        # Iterate through frames
        for frame_num in tqdm(range(num_save_frames)):
            
            # Read frame and make sure it's read in correctly
            ret, frame = vidread.read()
            if not ret:
                break
            
            # Plot on the frame if there is data to be used
            if self.xrpts is not None and self.ellipse_params is not None:
                
                try:
                    # Get out ellipse long/short axes and put into tuple
                    ellipse_axes = (int(self.ellipse_params.sel(                        \
                                        frame=frame_num,                                \
                                        ellipse_params='longaxis').values),             \
                                    int(self.ellipse_params.sel(                        \
                                        frame=frame_num,                                \
                                        ellipse_params='shortaxis').values))

                    # Get out ellipse phi and round to int
                    # Note: this is ellipse_phi not phi
                    ellipse_phi = int(np.rad2deg(self.ellipse_params.sel(
                                        frame=frame_num,
                                        ellipse_params='ellipse_phi').values))

                    # Get ellipse center out, round to int, and put into tuple
                    ellipse_cent = (int(self.ellipse_params.sel(
                                        frame=frame_num,
                                        ellipse_params='X0').values), 
                                    int(self.ellipse_params.sel(
                                        frame=frame_num,
                                        ellipse_params='Y0').values))
                    
                    # Update this frame with an ellipse
                    # ellipse plotted in blue
                    frame = cv2.ellipse(frame, ellipse_cent, ellipse_axes,
                                        ellipse_phi, 0, 360, (255,0,0), 2) 
                
                # Skip if the ell data from this frame are bad
                except (ValueError, KeyError):
                    pass

                try:
                    # iterate through each point in the list
                    for k in range(0, len(self.xrpts.isel(frame=frame_num)), 3):
                        # get the point center of each point num, k
                        pt_cent = (int(self.xrpts.isel(frame=frame_num,
                                                       point_loc=k).values),
                                   int(self.xrpts.isel(frame=frame_num,
                                                       point_loc=k+1).values))
                        # compare to threshold set in cfg and plot
                        if self.xrpts.isel(frame=frame_num, point_loc=k+2).values < self.cfg['Lthresh']:
                            # bad points in red
                            frame = cv2.circle(frame, pt_cent, 3, (0,0,255), -1)
                        elif self.xrpts.isel(frame=frame_num, point_loc=k+2).values >= self.cfg['Lthresh']:
                            # good points in green
                            frame = cv2.circle(frame, pt_cent, 3, (0,255,0), -1)
                
                except (ValueError, KeyError):
                    pass

            out_vid.write(frame)
        out_vid.release()


    def save_params(self):
        """ Save the NC file of parameters.
        """

        self.xrpts.name = self.camname+'_pts'
        self.xrframes.name = self.camname+'_video'
        self.ellipse_params.name = self.camname+'_ellipse_params'

        merged_data = [self.xrpts, self.ellipse_params, self.xrframes]

        if self.cfg['ridge_cyclotorsion']:

            self.rfit.name = self.camname+'_pupil_radius'
            self.shift.name = self.camname+'_omega'
            self.rfit_conv.name = self.camname+'_conv_pupil_radius'

            merged_data = merged_data + [self.rfit, self.shift, self.rfit_conv]

        self.safe_merge(merged_data)
        
        f_name = '{}_{}.nc'.format(self.recording_name, self.camname)

        savepath = os.path.join(self.recording_path, f_name)

        self.data.to_netcdf(savepath, engine='netcdf4',
                    encoding = {
                        self.camname+'_video': {"zlib": True,
                                                "complevel": 4}})

        print('Saved {}'.format(savepath))




    def process(self):
        """ Run eyecam preprocessing.
        """

        if self.cfg['run']['deinterlace']:
            self.deinterlace()

        elif not self.cfg['run']['deinterlace'] and (self.cfg['headcams_hflip'] or self.cfg['headcams_vflip']):
            self.flip_headcams()

        if self.cfg['fix_eyecam_contrast']:
            self.auto_contrast()

        if self.cfg['run']['pose_estimation']:
            self.pose_estimation()

        if self.cfg['run']['parameters']:
            
            self.gather_camera_files()
            self.pack_position_data()
            self.pack_video_frames()

            self.track_pupil()

            if self.cfg['ridge_cyclotorsion']:
                self.get_torsion_from_ridges()

            if self.cfg['write_diagnostic_videos']:
                self.eye_diagnostic_video()

            self.save_params()
