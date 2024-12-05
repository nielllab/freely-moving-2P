import os
import cv2
import subprocess
import pandas as pd
os.environ["DLClight"] = "True"
import deeplabcut

import fmr2e

def deinterlace(self, videos=None, timestamps=None,
                    exp_fps=30, quiet=False):
        """ Deinterlace videos and shift timestamps to match new video frames.

        If videos and timestamps are provided (as lists), only the provided
        filepaths will be processed. If lists are not provided, subdirectories
        will be searched within animal_directory in the options dictionary, config.
        Both videos and timestamps must be provided, for either to be used.

        Videos will also be rotated 180 deg (so that they are flipped in the horizontal
        and vertical directions) if the option is set in the config file.

        Parameters
        ----------
        videos : list
            List of eyecam and/or worldcam videos at 30fps (default is None). If
            the list is None, the subdirectories will be searched for videos.
        timestamps : list
            List of timestamp csv files for each video in videos (default is None).
            If the list is None, the subdirectories will be searched for timestamps.
        exp_fps : int
            Expected framerate of the videos (default is 30 Hz). If a video does not
            match this framerate, it will be skipped (e.g. if it has a frame rate of
            60 fps, it is assumed to have already been deinterlaced).
        quiet : bool
            When True, the function will not print status updates (default is False).
        
        """

        print('Deinterlacing {} video...'.format(self.camname))

        if 'EYE' in self.camname:

            if self.cfg['rotate_eyecam']:
                do_rotation = True
            else:
                do_rotation = False
        
        elif 'WORLD' in self.camname:

            if self.cfg['rotate_worldcam']:
                do_rotation = True
            else:
                do_rotation = False

        # search subdirectories if both lists are not given
        if videos is None or timestamps is None:
            
            videos = fmr2e.find('*{}*{}*.avi'.format(self.recording_name, self.camname),
                                                            self.recording_path)
            timestamps = fmr2e.find('*{}*{}*.csv'.format(self.recording_name, self.camname),
                                                            self.recording_path)
        
        # iterate through each video
        for vid in videos:

            current_path = os.path.split(vid)[0]

            # Make a save path that keeps the subdirectories. Get out a
            # key from the name of the video that will be shared with
            # all other data of this trial.
            
            vid_name = os.path.split(vid)[1]
            key_pieces = vid_name.split('.')[:-1]
            key = '.'.join(key_pieces)
            
            # open the video
            cap = cv2.VideoCapture(vid)
            
            # get some info about the video
            fps = cap.get(cv2.CAP_PROP_FPS) # frame rate

            if fps != exp_fps:
                return

            savepath = os.path.join(current_path, (key + 'deinter.avi'))
        
            if do_rotation:
                vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'
            elif not do_rotation:
                vf_val = 'yadif=1:-1:0, scale=640:480'

            # could add a '-y' after 'ffmpeg' and before ''-i' so that it overwrites
            # an existing file by default
            cmd = ['ffmpeg', '-i', vid, '-vf', vf_val, '-c:v', 'libx264',
                '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
                '256k']

            if self.cfg['allow_avi_overwrite'] is True:
                cmd.extend(['-y'])
            else:
                cmd.extend(['-n'])

            cmd.extend([savepath])
            
            if quiet is True:
                cmd.extend(['-loglevel', 'quiet'])

            subprocess.call(cmd)


def flip_headcams(self, quiet=True):
    """ Flip headcam videos horizontally and/or vertically.

    This function will flip headcam videos horizontally and/or vertically
    based on the options in the config file. This is only needed for videos
    that need to have their orientation changed but do not need to be
    deinterlaced.

    Parameters
    ----------
    quiet : bool
        When True, the function will not print status updates (default
        is True).
    """

    h = self.cfg['headcams_hflip']
    v = self.cfg['headcams_vflip']

    if h is True and v is True:
        vf_val = 'vflip, hflip'

    elif h is True and v is False:
        vf_val = 'hflip'

    elif h is False and v is True:
        vf_val = 'vflip'

    vid_list = fmr2e.find('*'+self.camname+'.avi', self.recording_path)

    for vid in vid_list:

        vid_name = os.path.split(vid)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)

        savepath = os.path.join(os.path.split(vid)[0], (key + 'deinter.avi'))

        cmd = ['ffmpeg', '-i', vid, '-vf', vf_val, '-c:v',
            'libx264', '-preset', 'slow', '-crf', '19',
            '-c:a', 'aac', '-b:a', '256k']

        if self.cfg['allow_avi_overwrite'] is True:
            cmd.extend(['-y'])
        else:
            cmd.extend(['-n'])

        cmd.extend([savepath])

        if quiet is True:
            cmd.extend(['-loglevel', 'quiet'])

        # Only do the rotation is at least one axis is being flipped
        if h is True or v is True:
            subprocess.call(cmd)



def batch_dlc_analysis(self, videos, project_cfg):
    """ Run DLC pose estimation on videos.

    Parameters
    ----------
    videos : str or list
        The path to the video file(s) to be analyzed.
    project_cfg : str
        The path to the project config file.

    """
    
    if isinstance(videos, str):
        videos = [videos]
    
    for vid in videos:
        
        if self.cfg['DLC_crop'] is True:
            deeplabcut.cropimagesandlabels(project_cfg,
                                            size=(400, 400),
                                            userfeedback=False)
        
        deeplabcut.analyze_videos(project_cfg, [vid])
        
        if self.cfg['DLC_filt'] is True:

            deeplabcut.filterpredictions(project_cfg, vid)


def pose_estimation(self):
    """ Run DLC pose estimation on the eye camera videos.
    """
    
    if self.camname in self.cfg['dlc_projects'].keys():
        cam_project = self.cfg['dlc_projects'][self.camname]
    else:
        cam_project = None

    if cam_project != '' and cam_project != 'None' and cam_project != None:
        
        # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
        if self.camname=='REYE' or self.camname=='LEYE':
            
            # find all the videos in the data directory that are from the current camera and are deinterlaced
            if self.cfg['strict_name'] is True:
                vids_this_cam = fmr2e.find('*'+self.camname+'*deinter.avi',  self.recording_path)
            
            elif self.cfg['strict_name'] is False:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
            
            # remove unflipped videos generated during jumping analysis
            bad_vids = fmr2e.find('*'+self.camname+'*unflipped*.avi', self.recording_path)
            
            for x in bad_vids:
                if x in vids_this_cam:
                    vids_this_cam.remove(x)
            ir_vids = fmr2e.find('*IR*.avi', self.recording_path)
            
            for x in ir_vids:
                if x in vids_this_cam:
                    vids_this_cam.remove(x)
            
            # warning for user if no videos found
            if len(vids_this_cam) == 0:
                print('no ' + self.camname + ' videos found -- maybe the videos are not deinterlaced yet?')
        
        else:
            
            # find all the videos for camera types that don't neeed to be deinterlaced
            if self.cfg['strict_name'] is True:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
            
            elif self.cfg['strict_name'] is False:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
        
        # analyze the videos with DeepLabCut
        # this gives the function a list of files that it will iterate over with the same DLC config file
        vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
        self.batch_dlc_analysis(vids2run, cam_project)






def pack_video_frames(self, usexr=True, dwnsmpl=None):
    """ Pack video frames into an array.

    Parameters
    ----------
    usexr : bool
        Whether to use xarray or not. Default is True.
    dwnsmpl : float
        How much to downsample the video frames. Default is None.

    Returns
    -------
    all_frames : np.ndarray
        Array of video frames. Only returned if usexr is False,
        otherwise returns None.

    """

    print('Packing video frames into array...')
    
    if dwnsmpl is None:
        dwnsmpl = self.cfg['video_dwnsmpl']
    
    # open the .avi file
    vidread = cv2.VideoCapture(self.video_path)
    
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*dwnsmpl),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*dwnsmpl)], dtype=np.uint8)
    
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0),
                            fx=dwnsmpl, fy=dwnsmpl,
                            interpolation=cv2.INTER_NEAREST)
        
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    
    if not usexr:
        return all_frames
    
    # store the combined video frames in an xarray
    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    
    # label frame numbers in the xarray
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    # delete all frames, since it's somewhat large in memory
    del all_frames
    self.xrframes = formatted_frames



def pack_position_data(self):
    """ Pack the camera's dlc points and timestamps together in one DataArray.
    """

    # check that pt_path exists
    if self.dlc_path is not None and self.dlc_path != [] and self.timestamp_path is not None:
        
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in self.camname and self.cfg['DLC_topMA'] is True:
            
            # add a step to convert pickle files here?
            pts = self.open_dlc_h5_multianimal()
        
        # otherwise, use regular h5 file read-in
        else:
            pts, self.pt_names = self.open_dlc_h5()
        
        # read time file, pass length of points so that it will know if that length matches the length of the timestamps
        # if they don't match because time was not interpolated to match deinterlacing, the timestamps will be interpolated
        time = self.read_timestamp_file(len(pts))
        
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        
        # label the camera view
        xrpts.name = self.camname
        
        # assign timestamps as a coordinate to the 
        try:
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
        
        # correcting for issue caused by small differences in number of frames
        except ValueError:
            
            diff = len(time[1:]) - len(xrpts['frame'])
            
            if diff > 0: # time is longer
                diff = abs(diff)
                new_time = time.copy()
                new_time = new_time[0:-diff]
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            
            elif diff < 0: # frame is longer
                diff = abs(diff)
                timestep = time[1] - time[0]
                new_time = time.copy()
                for i in range(1,diff+1):
                    last_value = new_time[-1] + timestep
                    new_time = np.append(new_time, pd.Series(last_value))
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            
            # equal (probably won't happen because ValueError should have been
            # caused by unequal lengths)
            else:
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
    
    # pt_path will have no data in it for world cam data, so it will make an
    # xarray with just timestamps
    elif self.dlc_path is None or self.dlc_path == [] and self.timestamp_path is not None:
        
        if self.timestamp_path is not None and self.timestamp_path != []:
            # read in the time
            if 'formatted' in self.timestamp_path:
                time = self.read_timestamp_file()
            else:
                time = self.read_timestamp_file(force_timestamp_shift=True)
            # setup frame indices
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            # assign frame coordinates, then timestamps
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
        
        elif self.timestamp_path is None or self.timestamp_path == []:
            xrpts = None

    # if timestamps are missing, still read in and format as xarray
    elif self.dlc_path is not None and self.dlc_path != [] and self.timestamp_path is None:
        
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in self.camname and self.cfg['DLC_topMA'] is True:
            # add a step to convert pickle files here?
            pts = self.open_dlc_h5_multianimal()
        
        # otherwise, use regular h5 file read-in
        else:
            pts, self.pt_names = self.open_dlc_h5()
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        # label the camera view
        xrpts.name = self.camname

    self.xrpts = xrpts