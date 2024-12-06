import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmr2e

class Topcam():

    def __init__(self, recording_path, recording_name, cfg=None):

        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            self.likelihood_thresh = 0.99
        elif cfg is not None:
            self.likelihood_thresh = cfg['likelihood_thresh']


    def get_head_body_yaw(self):
        pdf = PdfPages(os.path.join(self.recording_path,
                       (self.recording_name + '_' + self.camname + '_tracking_figs.pdf')))

        # Get timestamps
        topT = self.xrpts.timestamps.copy()
        topT = topT - topT[0]

        # Plot traces of each labeled point and show frequency of good tracking
        pt_names = list(self.xrpts['point_loc'].values)
        x_cols = [i for i in pt_names if '_x' in i]
        y_cols = [i for i in pt_names if '_y' in i]

        plt.subplots(int(np.ceil(len(pt_names)/9)), 3,
                     figsize=(20,15))

        for i in range(len(x_cols)):

            x = self.xrpts.sel(point_loc=x_cols[i])
            y = self.xrpts.sel(point_loc=y_cols[i])

            plt.subplot(int(np.ceil(len(pt_names)/9)), 3, i+1)
            plt.plot(x)
            plt.plot(y)

            frac_good = np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)

            plt.title(pt_names[::3][i][:-2] + ' good='+str(np.round(frac_good.values,4)))

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # plastic or metal corners?
        ctypes = ['m','p']
        frac_good = np.zeros(2)

        for ctypenum in range(2):

            ctype = ctypes[ctypenum]
            fg = 0

            for cpos in ['tl','tr','br','bl']:

                cname = cpos+'_'+ctype+'_corner_'
                x = self.xrpts.sel(point_loc=cname+'x')
                y = self.xrpts.sel(point_loc=cname+'y')
                fg += np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)

            frac_good[ctypenum] = fg

        metal_corners = np.argmax(frac_good)==0

        # Define cm using top or bottom corners?
        frac_good = np.zeros(2)
        m_or_p = ('m' if metal_corners else 'p')
        
        for c, poslet in enumerate(['t','b']):
            
            fg = 0
            for lrpos in ['l','r']:
                cname = poslet+lrpos+'_'+m_or_p+'_corner_'
                x = self.xrpts.sel(point_loc=cname+'x')
                y = self.xrpts.sel(point_loc=cname+'y')
                fg += np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)
            frac_good[c] = fg

        use_top_for_dist = np.argmax(frac_good)==0

        # Conversion from pixels to cm
        tb = ('t' if use_top_for_dist else 'b')
        mp = ('m' if metal_corners else 'p')
        left = tb+'l_'+mp+'_corner_x'
        right = tb+'r_'+mp+'_corner_x'
        
        dist_pxls = np.nanmedian(self.xrpts.sel(point_loc=right)) -                     \
                            np.nanmedian(self.xrpts.sel(point_loc=left))
        
        pxls2cm = dist_pxls / self.dist_cm

        if np.isnan(pxls2cm):
            pxls2cm = self.default_pxls2cm

        # Topdown speed using neck point
        smooth_x = fmr2e.convfilt(fmr2e.nanmedfilt(self.xrpts.sel(                              \
                                point_loc='center_neck_x').values, 7).squeeze(), box_pts=20)
        
        smooth_y = fmr2e.convfilt(fmr2e.nanmedfilt(self.xrpts.sel(                              \
                                point_loc='center_neck_y').values, 7).squeeze(), box_pts=20)
        
        top_speed = np.sqrt(np.diff((smooth_x*60) / pxls2cm)**2 + np.diff((smooth_y*60) / pxls2cm)**2)

        top_speed[top_speed>25] = np.nan

        plt.figure()
        plt.plot(topT[:-1], top_speed, linewidth=1)
        plt.xlabel('sec')
        plt.ylabel('cm/sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.hist(top_speed, bins=40, density=True)
        plt.xlabel('cm/sec')
        plt.ylabel('fraction of time')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Get head angle from ear points
        lear_x = fme.nanmedfilt(self.xrpts.sel(point_loc='left_ear_x').values, 7).squeeze()
        lear_y = fme.nanmedfilt(self.xrpts.sel(point_loc='left_ear_y').values, 7).squeeze()
        rear_x = fme.nanmedfilt(self.xrpts.sel(point_loc='right_ear_x').values, 7).squeeze()
        rear_y = fme.nanmedfilt(self.xrpts.sel(point_loc='right_ear_y').values, 7).squeeze()

        # Rotate 90deg because ears are perpendicular to head yaw
        head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)

        head_yaw_deg = np.rad2deg(head_yaw % (2*np.pi))


        plt.figure(figsize=(15,5))
        plt.plot(topT, head_yaw_deg, '.', markersize=1)
        plt.ylabel('head yaw (deg)')
        plt.xlabel('sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Body angle from neck and back points
        neck_x = fme.nanmedfilt(self.xrpts.sel(point_loc='center_neck_x').values, 7).squeeze()
        neck_y = fme.nanmedfilt(self.xrpts.sel(point_loc='center_neck_y').values, 7).squeeze()
        back_x = fme.nanmedfilt(self.xrpts.sel(point_loc='center_haunch_x').values, 7).squeeze()
        back_y = fme.nanmedfilt(self.xrpts.sel(point_loc='center_haunch_y').values, 7).squeeze()
        
        body_yaw = np.arctan2((neck_y - back_y), (neck_x - back_x))
        body_yaw_deg = np.rad2deg(body_yaw % (2*np.pi))

        plt.figure(figsize=(15,5))
        plt.plot(topT, body_yaw_deg, '.', markersize=1)
        plt.ylabel('body yaw (deg)')
        plt.xlabel('sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        body_head_diff = head_yaw - body_yaw

        body_head_diff[body_head_diff<-np.deg2rad(120)] = np.nan
        body_head_diff[body_head_diff>np.deg2rad(120)] = np.nan

        plt.figure(figsize=(15,5))
        plt.plot(topT, np.rad2deg(body_head_diff), '.', markersize=1)
        plt.ylabel('head yaw - body yaw (deg)')
        plt.xlabel('sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Angle of body movement ("movement yaw")
        x_disp = np.diff((smooth_x*60) / pxls2cm)
        y_disp = np.diff((smooth_y*60) / pxls2cm)

        movement_yaw = np.arctan2(y_disp, x_disp)
        movement_yaw_deg = np.rad2deg(movement_yaw % (2*np.pi))

        plt.figure(figsize=(15,5))
        plt.plot(topT[:-1], movement_yaw_deg, '.', markersize=1)
        plt.ylabel('animal yaw (deg)')
        plt.xlabel('sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(15,5))
        plt.plot(topT[:-1], movement_yaw_deg - body_yaw_deg[:-1], '.', markersize=1)
        plt.ylabel('movement yaw - body yaw (deg)')
        plt.xlabel('sec')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Definitions of state
        movement_minus_body = movement_yaw - body_yaw[:-1]

        running = (top_speed>self.running_thresh)
        forward = (np.abs(movement_minus_body) < np.deg2rad(self.forward_thresh))
        backward = (np.abs(movement_minus_body + np.deg2rad(180) % (2*np.pi)) <             \
                                    np.deg2rad(self.forward_thresh))
        
        forward_run = running * forward
        backward_run = running * backward
        fine_motion = running * ~forward * ~backward
        immobility = ~running

        # Plot of movement, heading, and speed
        if self.make_all_plots:

            xbounds = np.array([np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_x').values),
                                np.nanmedian(self.xrpts.sel(point_loc='tr_p_corner_x').values),
                                np.nanmedian(self.xrpts.sel(point_loc='br_p_corner_x').values),
                                np.nanmedian(self.xrpts.sel(point_loc='bl_p_corner_x').values),
                                np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_x').values)])
            
            ybounds = np.array([np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_y').values),
                                np.nanmedian(self.xrpts.sel(point_loc='tr_p_corner_y').values),
                                np.nanmedian(self.xrpts.sel(point_loc='br_p_corner_y').values),
                                np.nanmedian(self.xrpts.sel(point_loc='bl_p_corner_y').values),
                                np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_y').values)])
            
            start = 1000
            maxf = 3600

            plt.subplots(1,2,figsize=(15,6))
            plt.subplot(121)
            plt.plot(xbounds, ybounds, 'k-')

            cmap = plt.cm.jet(np.linspace(0,1,maxf))

            for f in range(start, start+maxf):
                plt.plot(neck_x[f], neck_y[f], '.', color=cmap[f-start])

            plt.subplot(122)
            plt.plot(xbounds, ybounds, 'k-')

            speed_bins = np.linspace(0,int(np.ceil(np.nanmax(top_speed)/3)))
            spdcolors = plt.cm.magma(speed_bins)

            for f in np.arange(start,start+maxf+10,10):

                if ~np.isnan(top_speed[f]):
                    usecolor = spdcolors[np.argmin(np.abs(top_speed[f] - speed_bins))]
                
                else:
                    continue

                x0 = neck_x[f]
                y0 = neck_y[f]

                dX = 15*np.cos(head_yaw[f])
                dY = 15*np.sin(head_yaw[f])

                plt.arrow(x0, y0, dX, dY, facecolor=usecolor, width=7, edgecolor='k')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        pdf.close()

        if self.cfg['write_diagnostic_videos'] and self.make_speed_yaw_video:

            vid_save_path = os.path.join(self.recording_path,
                                (self.recording_name+'_'+self.camname+'_speed_yaw.avi'))
            
            start = 1000
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_vid = cv2.VideoWriter(vid_save_path, fourcc, 60.0, (432, 288))
            maxprev = 25

            for f in tqdm(range(start,start+3600)):

                fig = plt.figure()

                plt.imshow(self.xrframes[f,:,:].astype(np.uint8), cmap='gray')
                plt.ylim([135,0])
                plt.xlim([0,180])
                plt.axis('off')

                plt.plot(lear_x[f]*0.25, lear_y[f]*0.25, 'b*')
                plt.plot(rear_x[f]*0.25, rear_y[f]*0.25, 'b*')

                plt.plot([neck_x[f]*0.25, (neck_x[f]*0.25)+15*np.cos(head_yaw[f])],
                         [neck_y[f]*0.25,(neck_y[f]*0.25)+15*np.sin(head_yaw[f])],
                         '-', linewidth=2, color='cyan') # head yaw
                
                plt.plot([back_x[f]*0.25, (back_x[f]*0.25)-15*np.cos(body_yaw[f])],
                         [back_y[f]*0.25, (back_y[f]*0.25)-15*np.sin(body_yaw[f])],
                         '-', linewidth=2, color='pink') # body yaw
                
                for p in range(maxprev):

                    prevf = f - p

                    plt.plot(neck_x[prevf]*0.25,
                             neck_y[prevf]*0.25, 'o', color='tab:purple',
                             alpha=(maxprev-p)/maxprev) # neck position history
                    
                # arrow for vector of motion
                if forward_run[f]:
                    movvec_color = 'tab:green'
                elif backward_run[f]:
                    movvec_color = 'tab:orange'
                elif fine_motion[f]:
                    movvec_color = 'tab:olive'
                elif immobility[f]:
                    movvec_color = 'tab:red'
                
                plt.arrow(neck_x[f]*0.25, neck_y[f]*0.25,
                          x_disp[f]*3, y_disp[f]*3,
                          color=movvec_color, width=1)
                
                # Save the frame out
                fig.canvas.draw()
                frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()

                img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
                out_vid.write(img.astype('uint8'))

            out_vid.release()
        
        # Collect data into xarray
        # start by adding properties to save in a list
        prop_dict = {
            'speed': top_speed,
            'head_yaw': head_yaw,
            'body_yaw': body_yaw,
            'body_head_diff': body_head_diff,
            'movement_yaw': movement_yaw,
            'movement_minus_body': movement_minus_body,
            'forward_run': forward_run,
            'backward_run': backward_run,
            'fine_motion': fine_motion,
            'immobility': immobility
        }

        # Pad NaNs onto end of properties that are shorted than
        # frame count (because of np.diff)
        target_sz = len(lear_x)

        for key, val in prop_dict.items():

            sz = np.size(val,0)
            if sz < target_sz:

                addto = np.zeros([target_sz - sz])
                addto[:] = np.nan
                newprop = np.concatenate([val, addto],0)
                prop_dict[key] = newprop

        # Pack values into array
        prop_arr = np.zeros([len(prop_dict.keys()), len(lear_x)])

        for count, val in enumerate(prop_dict.values()):
            prop_arr[count,:] = val

        self.xrprops = xr.DataArray(prop_arr.T,
                       coords = [('frame', range(len(lear_x))),
                                 ('prop', list(prop_dict.keys()))],
                       dims = ['frame', 'prop']
                       )