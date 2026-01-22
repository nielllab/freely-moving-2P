# -*- coding: utf-8 -*-
"""
Boundary tuning analysis tools for freely-moving 2P experiments.

This module provides the `BoundaryTuning` class and supporting functions for calculating
rate maps, occupancy, and classifying boundary cells (EBC/IEBC) in rodent navigation experiments.

Classes:
    BoundaryTuning: Main class for boundary cell analysis.

Functions:
    convert_bools_to_ints(data): Recursively convert bools in a dict to ints (for HDF5 saving).
    rate_map_mp(...): Multiprocessing helper for rate map calculation.
    calc_MRL_mp(...): Multiprocessing helper for mean resultant length.
    calc_shfl_mean_resultant_mp(...): Multiprocessing helper for shuffled MRL.

Example usage:
    >>> from fm2p.utils.boundary_tuning import BoundaryTuning
    >>> bt = BoundaryTuning(preprocessed_data)
    >>> bt.identify_responses(use_angle='head')
    >>> print(bt.data_out['is_EBC'])

Author: DMM, last modified Oct 2025
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew
from scipy.ndimage import label
from tqdm import tqdm
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

import fm2p


def convert_bools_to_ints(data):
    """
    Recursively convert all boolean values in a dictionary to integers.
    Useful for saving data to HDF5, which does not support bools.

    Parameters
    ----------
    data : dict
        Input dictionary (possibly nested).

    Returns
    -------
    dict
        Dictionary with all bools replaced by ints.
    """

    new_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_dict[key] = convert_bools_to_ints(value)  # recursive call
        elif (isinstance(value, bool)) or (isinstance(value, np.bool_)):
            new_dict[key] = int(value)
        elif (isinstance(value, np.complex128)):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict


def rate_map_mp(spike_rate, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size):
    """
    Calculate a 2D rate map for a single cell using spike rates and occupancy.
    Used for multiprocessing.

    Parameters
    ----------
    spike_rate : np.ndarray
        1D array of spike rates for each frame.
    occupancy : np.ndarray
        2D array of occupancy counts (angular x distance bins).
    ray_distances : np.ndarray
        2D array of distances to wall for each frame and angle.
    ray_width : float
        Width of each angular bin (degrees).
    dist_bin_edges : np.ndarray
        1D array of distance bin edges (cm).
    dist_bin_size : float
        Size of each distance bin (cm).

    Returns
    -------
    rate_map : np.ndarray
        2D array (angular x distance) of firing rates.
    """

    N_angular_bins = int(360 / ray_width)
    N_distance_bins = len(dist_bin_edges) - 1

    rate_map = np.zeros((N_angular_bins, N_distance_bins))

    for f in range(len(spike_rate)):
        for a, ang in enumerate(np.arange(0, 360, ray_width)):
            for d, dist_bin_start in enumerate(dist_bin_edges[:-1]):
                dist_bin_end = dist_bin_start + dist_bin_size
                if (ray_distances[f, a] >= dist_bin_start) and (ray_distances[f, a] < dist_bin_end):
                    rate_map[a, d] += spike_rate[f]

    rate_map /= occupancy + 1e-6  # avoid division by zero

    return rate_map


def calc_MRL_mp(ratemap, ray_width, dist_bin_cents):
    """
    Calculate the mean resultant length (MRL) of a 2D rate map.
    Used for multiprocessing.

    Parameters
    ----------
    ratemap : np.ndarray
        2D array (angular x distance) of firing rates.
    ray_width : float
        Width of each angular bin (degrees).
    dist_bin_cents : np.ndarray
        1D array of distance bin centers (cm).

    Returns
    -------
    mean_resultant_length : float
        The mean resultant length of the rate map.
    """

    N_angular_bins, N_distance_bins = ratemap.shape
    angs_rad = np.deg2rad(np.arange(0, 360, ray_width))

    # create a meshgrid of angles and distances
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')

    # calculate the mean resultant vector
    mr = np.sum(ratemap * np.exp(1j * angs_mesh)) / (N_angular_bins * N_distance_bins)
    
    mean_resultant_length = np.abs(mr)

    return mean_resultant_length


def calc_shfl_mean_resultant_mp(spikes, useinds, occupancy, ray_distances, ray_width,
                             dist_bin_edges, dist_bin_size, dist_bin_cents, is_IEBC):
    """
    Calculate the mean resultant length (MRL) for a shuffled spike train.
    Used for multiprocessing shuffling.

    Parameters
    ----------
    spikes : np.ndarray
        1D array of spike counts.
    useinds : np.ndarray
        Boolean mask of frames to use.
    occupancy : np.ndarray
        2D array of occupancy counts.
    ray_distances : np.ndarray
        2D array of distances to wall.
    ray_width : float
        Width of each angular bin (degrees).
    dist_bin_edges : np.ndarray
        1D array of distance bin edges (cm).
    dist_bin_size : float
        Size of each distance bin (cm).
    dist_bin_cents : np.ndarray
        1D array of distance bin centers (cm).
    is_IEBC : bool
        Whether the cell is an inverse EBC (invert the map).

    Returns
    -------
    shf_mrl : float
        Mean resultant length for the shuffled map.
    """

    N_frames = np.sum(useinds)

    shift_amount = np.random.randint(int(0.1*N_frames), int(0.9*N_frames))
    shifted_spikes = np.roll(spikes[useinds], shift_amount)
    shifted_ratemap = rate_map_mp(shifted_spikes, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size)
    if is_IEBC:
        shifted_ratemap = np.max(shifted_ratemap) - shifted_ratemap + np.min(shifted_ratemap)
    shf_mrl = calc_MRL_mp(shifted_ratemap, ray_width, dist_bin_cents)

    return shf_mrl


class BoundaryTuning:
    """
    Main class for boundary cell analysis in freely-moving 2P experiments.

    Methods
    -------
    calc_allo_yaw(): Compute allocentric head and pupil angles.
    calc_ego(): Compute egocentric angles.
    get_ray_distances(angle): Compute distances to wall for each frame and angle.
    calc_occupancy(inds): Compute occupancy map for given indices.
    calc_rate_maps_mp(): Compute rate maps using multiprocessing.
    calc_rate_maps(use_mp): Compute rate maps (optionally with multiprocessing).
    smooth_rate_maps(): Smooth all rate maps for visualization.
    smooth_map_pair(map1, map2): Smooth two arbitrary rate maps.
    identify_inverse_responses(): Classify inverse EBCs (IEBCs).
    identify_boundary_cells(): Classify EBCs using split-half and MRL criteria.
    identify_responses(...): Full pipeline for boundary cell analysis.
    save_results(savepath): Save results to HDF5.
    """

    def __init__(self, preprocessed_data):
        """
        Initialize BoundaryTuning object with preprocessed data.

        Parameters
        ----------
        preprocessed_data : dict
            Dictionary containing all required behavioral and spike data.
        """

        self.data = preprocessed_data

        self.ray_width = 3 # deg
        self.max_dist = 26 # cm
        self.dist_bin_size = 2. # cm

        self.head_ang = None
        self.pupil_ang = None

        self.criteria_out = {}
        for c in range(np.size(self.data['norm_spikes'],0)):
            self.criteria_out['cell_{:03d}'.format(c)] = {}
    
    def calc_allo_yaw(self):
        """
        Compute allocentric head and pupil angles from preprocessed data.
        Sets self.head_ang and self.pupil_ang.
        """
        self.head_ang = self.data['head_yaw_deg']

    def calc_allo_pupil(self):
        # i do NOT want to use retinocentric pillar location
        # instead, use gaze orientation

        self.pupil_ang = self.data['head_yaw_deg'].copy()[:-1] + self.data['pupil_from_head'].copy()
        self.pupil_ang = self.pupil_ang % 360

        # self.pupil_ang = self.data['retinocentric'] + 180.


    def calc_ego(self):
        """
        Compute egocentric angles from preprocessed data.
        Sets self.ego_ang.
        """
        self.ego_ang = self.data['egocentric'] + 180.

    def get_ray_distances(self, angle='head'):
        """
        Compute the distance from the animal to the closest wall for each frame and angle.

        Parameters
        ----------
        angle : str, optional
            Which angle to use ('head', 'pupil', or 'ego'). Default is 'head'.

        Returns
        -------
        ray_distances : np.ndarray
            2D array (frames x angles) of distances to the closest wall.
        """

        # Select the angle trace based on the requested reference frame
        if angle == 'egow':
            if self.head_ang is None:
                self.calc_allo_yaw()
            angle_trace = self.head_ang
        elif angle == 'pupil':
            if self.pupil_ang is None:
                self.calc_allo_pupil()
            angle_trace = self.pupil_ang
        elif angle == 'egop':
            if self.pupil_ang is None:
                self.calc_ego()
            angle_trace = self.ego_ang
        elif angle == 'retino':
            angle_trace = self.data['retinocentric'] + 180.

        x_trace = self.data['head_x'].copy() / self.data['pxls2cm']
        y_trace = self.data['head_y'].copy() / self.data['pxls2cm']

        use_inds = np.where(self.useinds)[0]
        N_frames = len(use_inds)

        x_trace = x_trace[use_inds]
        y_trace = y_trace[use_inds]

        # Why was this length check added? Currently causing IndexError... commented out 9/18/25
        # if len(use_inds) > len(angle_trace):
        #     angle_trace = np.append(angle_trace, angle_trace[-1])
        # elif len(use_inds) < len(angle_trace):
        #     angle_trace = angle_trace[:-1]

        angle_trace = angle_trace[use_inds]
        angle_trace = np.deg2rad(angle_trace)

        # Use the actual arena corners (in cm) for wall definitions
        BL = (self.data['arenaBL']['x'] / self.data['pxls2cm'], self.data['arenaBL']['y'] / self.data['pxls2cm'])
        BR = (self.data['arenaBR']['x'] / self.data['pxls2cm'], self.data['arenaBR']['y'] / self.data['pxls2cm'])
        TR = (self.data['arenaTR']['x'] / self.data['pxls2cm'], self.data['arenaTR']['y'] / self.data['pxls2cm'])
        TL = (self.data['arenaTL']['x'] / self.data['pxls2cm'], self.data['arenaTL']['y'] / self.data['pxls2cm'])

        wall_entries = [
            [BL[0], BL[1], BR[0], BR[1]],  # Bottom wall
            [BR[0], BR[1], TR[0], TR[1]],  # Right wall
            [TR[0], TR[1], TL[0], TL[1]],  # Top wall
            [TL[0], TL[1], BL[0], BL[1]]   # Left wall
        ]

        rays_rad = np.zeros((N_frames, int(360 / self.ray_width)))
        for f in range(N_frames):
            for r, ang in enumerate(np.arange(0, 360, self.ray_width)):
                rays_rad[f,r] = angle_trace[f] + np.deg2rad(ang)
        
        self.ray_distances = np.zeros([
            np.size(rays_rad,0),
            int(360 / self.ray_width)
        ]) * np.nan

        for fr in tqdm(range(np.size(rays_rad,0))):

            ray_distances = []
        
            for ri in range(np.size(rays_rad,1)):

                # Get distance to closest wall for each ray

                intersections = []
                closest_walls = []
                
                ray_ang = rays_rad[fr, ri]

                ray_vec = np.vstack((
                    np.cos(ray_ang), # x
                    np.sin(ray_ang)  # y
                ))

                for wall in wall_entries:
                    start = np.array([wall[0], wall[1]])
                    end = np.array([wall[2], wall[3]])
                    vector = end - start

                    # calculate the determinant (if 0, lines are parallel, no intersection)
                    det = np.cross(vector, ray_vec.T)
                    if any(det == 0):
                        continue

                    # calculate the relative position of ray origin to wall start point
                    relative_pos = np.array([x_trace[fr], y_trace[fr]]) - start

                    # calculate how far along the wall line the intersection occurs
                    # (t = 0 -> wall.start; t = 1 -> wall.end)
                    t = np.cross(relative_pos, ray_vec.T) / det
                    # if t is not between 0 and 1, the intersection is outside the finite wall line
                    if np.all((t < 0) | (t > 1)):
                        continue  # skip

                    # after these checks are passed, calculate the intersection coordinates
                    intersection = (start + np.outer(t, vector)).flatten()

                    # check if the intersection is really in the direction of the ray (from current frame position)
                    to_intersection = intersection - np.array([x_trace[fr], y_trace[fr]])
                    if np.dot(to_intersection, ray_vec.flatten()) < 0:
                        continue

                    intersections.append(intersection)
                    # calculate Euclidean distance from (x, y) to the intersection
                    distance = np.linalg.norm(intersection - np.array([x_trace[fr], y_trace[fr]]))
                    closest_walls.append(distance)

                if len(closest_walls)==0:
                    min_dist = np.nan
                    ray_distances.append(min_dist)
                else:
                    min_dist = np.nanmin(closest_walls) # distance of closest wall for that ray
                    ray_distances.append(min_dist) # append that distance to bin distance list
            
            self.ray_distances[fr,:] = np.array(ray_distances)

        # calculate distance bin edges
        self.dist_bin_edges = np.arange(0, self.max_dist + self.dist_bin_size, self.dist_bin_size)
        # calculate distance bin center positions
        self.dist_bin_cents = self.dist_bin_edges[:-1] + (self.dist_bin_size / 2)

        return self.ray_distances
    
    def calc_occupancy(self, inds=None):
        """
        Calculate the occupancy map (number of samples per angular x distance bin).

        Parameters
        ----------
        inds : array-like or None, optional
            Indices of frames to include. If None, use all frames.
            Can be integer indices or a boolean mask.

        Returns
        -------
        occupancy : np.ndarray
            2D array (angular x distance) of occupancy counts.
        """

        if np.size(self.ray_distances, 0) > (np.where(inds)[0]).size:
            kept_indices = np.nonzero(self.useinds)[0]
            mask_in_target = np.isin(kept_indices, inds)
            ray_distances = self.ray_distances.copy()[mask_in_target,:]
        else:
            ray_distances = self.ray_distances.copy()

        assert np.size(ray_distances, 0) == (np.where(inds)[0]).size

        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1
        occupancy = np.zeros((N_angular_bins, N_distance_bins))
        
        for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
            dist_bin_end = dist_bin_start + self.dist_bin_size
            
            # create a mask of where the distance falls within the current distance bin
            mask = (ray_distances >= dist_bin_start) & (ray_distances < dist_bin_end)

            # sum across frames to get occupancy for each angular bin
            occupancy[:, d] = np.sum(mask, axis=0)
        return occupancy
    
    def calc_rate_maps_mp(self):
        """
        Calculate rate maps for all cells using multiprocessing for speed.

        Returns
        -------
        rate_maps : np.ndarray
            3D array (cells x angular x distance) of firing rates.
        """
        nCells = np.size(self.data['norm_spikes'], 0)
        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        n_proc = multiprocessing.cpu_count() - 1
        print('  -> Starting multiprocessing pool (n_workers: {}/{}).'.format(n_proc, multiprocessing.cpu_count()))
        pbar = tqdm(total=nCells)

        def update_progress_bar(*args):
            pbar.update()

        spikes = self.data['norm_spikes'].copy()[:, self.useinds]

        pool = multiprocessing.Pool(processes=n_proc)

        mp_param_set = [
            pool.apply_async(
                rate_map_mp,
                args=(
                    spikes[cell_num, :],
                    self.occupancy,
                    self.ray_distances,
                    self.ray_width,
                    self.dist_bin_edges,
                    self.dist_bin_size
                ),
                callback=update_progress_bar
            ) for cell_num in range(nCells)
        ]
        mp_outputs = [result.get() for result in mp_param_set]

        self.rate_maps = np.zeros((nCells, N_angular_bins, N_distance_bins))

        for c, rmap in enumerate(mp_outputs):
            self.rate_maps[c, :, :] = rmap

        pbar.close()
        pool.close()

        return self.rate_maps
    
    def calc_rate_maps(self, use_mp=True):
        """
        Calculate rate maps for all cells.
        Optionally use multiprocessing for speed.

        Parameters
        ----------
        use_mp : bool, optional
            Whether to use multiprocessing (default True).

        Returns
        -------
        rate_maps : np.ndarray
            3D array (cells x angular x distance) of firing rates.
        """
        if use_mp is True:
            return self.calc_rate_maps_mp()

        spikes = self.data['norm_spikes'].copy()[:, self.useinds]

        N_cells = self.data['norm_spikes'].shape[0]
        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        self.rate_maps = np.zeros((N_cells, N_angular_bins, N_distance_bins))

        for c in tqdm(range(N_cells)):
            spike_rate = spikes[c, :]
            for f in range(len(spike_rate)):
                for a, ang in enumerate(np.arange(0, 360, self.ray_width)):
                    for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
                        dist_bin_end = dist_bin_start + self.dist_bin_size
                        if (self.ray_distances[f, a] >= dist_bin_start) and (self.ray_distances[f, a] < dist_bin_end):
                            self.rate_maps[c, a, d] += spike_rate[f]

            self.rate_maps[c, :, :] /= self.occupancy + 1e-6  # avoid division by zero

        return self.rate_maps
    
    def smooth_rate_maps(self):
        """
        Smooth all rate maps using a Gaussian filter for visualization.
        Handles angular wraparound by padding.

        Returns
        -------
        smoothed_rate_maps : np.ndarray
            3D array (cells x angular x distance) of smoothed rates.
        """
        smoothed_rate_maps = self.rate_maps.copy()

        for c in range(self.rate_maps.shape[0]):
            # pad the rate map by concatenating three copies along the angular axis
            temp_padded = np.vstack((smoothed_rate_maps[c, :, :], smoothed_rate_maps[c, :, :], smoothed_rate_maps[c, :, :]))
            # smooth the padded rate map
            # TODO: try sigma shapes that are not symetric (i.e., smooth more along angles than i do along distance axis)
            temp_smoothed = gaussian_filter(temp_padded, sigma=1)

            # slice the middle third to get the final smoothed rate map
            smoothed_rate_maps[c, :, :] = temp_smoothed[smoothed_rate_maps.shape[1]:2*smoothed_rate_maps.shape[1], :]

        self.smoothed_rate_maps = smoothed_rate_maps
        return self.smoothed_rate_maps

    def smooth_map_pair(self, map1, map2):
        """
        Smooth two arbitrary rate maps using the same logic as smooth_rate_maps,
        but without modifying self.rate_maps.

        Parameters
        ----------
        map1, map2 : np.ndarray
            2D rate maps to smooth (angular x distance).

        Returns
        -------
        smoothed_map1, smoothed_map2 : np.ndarray
            Smoothed versions of the input maps.
        """
        from scipy.ndimage import gaussian_filter
        smoothed_maps = []
        for ratemap in [map1, map2]:
            temp_padded = np.vstack((ratemap, ratemap, ratemap))
            temp_smoothed = gaussian_filter(temp_padded, sigma=1)
            smoothed = temp_smoothed[ratemap.shape[0]:2*ratemap.shape[0], :]
            smoothed_maps.append(smoothed)
        return smoothed_maps[0], smoothed_maps[1]
    
    def _invert_ratemap(self, ratemap):
        """
        Invert a rate map (for IEBC classification).
        Returns max - map + min.
        """
        return np.max(ratemap) - ratemap + np.min(ratemap)


    def _measure_skewness(self, ratemap):
        """
        Compute skewness of the rate map and test if it is negative.
        Returns skew value and pass/fail boolean.
        """
        skew_val = skew(ratemap.flatten())
        passes = skew_val < 0.
        return skew_val, passes

    def _calc_dispersion(self, ratemap):
        """
        Calculate the spatial dispersion of the top 10% bins in the rate map.
        Returns mean distance from centroid.
        """
        N_angular_bins, N_distance_bins = ratemap.shape
        angs_rad = np.deg2rad(np.arange(0, 360, self.ray_width))
        dist_cents = self.dist_bin_cents

        x_coords = np.zeros((N_angular_bins, N_distance_bins))
        y_coords = np.zeros((N_angular_bins, N_distance_bins))

        # fill in the x and y coordinates
        for a in range(N_angular_bins):
            for d in range(N_distance_bins):
                x_coords[a, d] = dist_cents[d] * np.cos(angs_rad[a])
                y_coords[a, d] = dist_cents[d] * np.sin(angs_rad[a])

        # identify bins in the top 10% of the map
        threshold = np.percentile(ratemap, 90)
        top_bins = ratemap >= threshold

        # get the coordinates of the top bins
        top_x = x_coords[top_bins]
        top_y = y_coords[top_bins]

        if len(top_x) < 2:
            return np.inf  # not enough points to calculate dispersion

        # calculate the centroid of the top bins
        centroid_x = np.mean(top_x)
        centroid_y = np.mean(top_y)

        # calculate the dispersion as the mean distance from the centroid
        distances = np.sqrt((top_x - centroid_x)**2 + (top_y - centroid_y)**2)
        dispersion = np.mean(distances)

        return dispersion
    
    def _measure_dispursion(self, ratemap):
        """
        Compare dispersion of normal and inverted rate maps.
        Returns both dispersions and pass/fail boolean.
        """
        normal_dispersion = self._calc_dispersion(ratemap)
        inv_ratemap = self._invert_ratemap(ratemap)
        inverted_dispersion = self._calc_dispersion(inv_ratemap)

        passes = inverted_dispersion < normal_dispersion

        return normal_dispersion, inverted_dispersion, passes
    
    def _calc_receptive_field_size(self, ratemap):
        """
        Calculate the size of the largest connected component above median in the rate map.
        Returns fraction of total map size.
        """
        # pad the angular axis with first and last columns to avoid edge effects
        padded_ratemap = np.vstack((ratemap[-1,:], ratemap, ratemap[0,:]))
        threshold = np.percentile(padded_ratemap, 50)
        binary_map = padded_ratemap >= threshold

        structure = np.ones((3,3))
        labeled_array, num_features = label(binary_map, structure=structure)
    
        if num_features == 0:
            return
        
        # remove the wrap around padding
        labeled_array = labeled_array[1:-1, :]
        
        # find the largest connected component
        largest_cc_size = 0
        for i in range(1, num_features + 1):
            cc_size = np.sum(labeled_array == i)
            if cc_size > largest_cc_size:
                largest_cc_size = cc_size

        receptive_field_size = largest_cc_size / ratemap.size  # percent of total map size

        return receptive_field_size
    
    def _measure_receptive_field_size(self, ratemap):
        """
        Compare receptive field size of normal and inverted rate maps.
        Returns both sizes and pass/fail boolean.
        """
        normal_rf_size = self._calc_receptive_field_size(ratemap)
        inv_ratemap = self._invert_ratemap(ratemap)
        inverted_rf_size = self._calc_receptive_field_size(inv_ratemap)

        passes = inverted_rf_size < normal_rf_size

        return normal_rf_size, inverted_rf_size, passes
    
    def identify_inverse_responses(self, inv_criteria_thresh=2):
        """
        Classify cells as inverse EBCs (IEBCs) based on skewness, dispersion, and receptive field size.

        Parameters
        ----------
        inv_criteria_thresh : int, optional
            Number of criteria that must be met to classify as IEBC (default 2).

        Returns
        -------
        is_IEBC : np.ndarray
            Boolean array indicating which cells are IEBCs.
        """

        N_cells = self.rate_maps.shape[0]
        self.is_IEBC = np.zeros(N_cells, dtype=bool)

        for c in tqdm(range(N_cells)):
            ratemap = self.rate_maps[c, :, :]

            skew_val, skew_pass = self._measure_skewness(ratemap)

            normal_dispersion, inverted_dispersion, disp_pass = self._measure_dispursion(ratemap)

            normal_rf_size, inverted_rf_size, rf_pass = self._measure_receptive_field_size(ratemap)

            pass_count = sum([skew_pass, disp_pass, rf_pass])
            if pass_count >= inv_criteria_thresh:
                self.is_IEBC[c] = True

            temp_dict = {
                'skewness_val': skew_val,
                'skewness_pass': int(skew_pass),
                'dispersion_inverted': inverted_dispersion,
                'dispersion_normal': normal_dispersion,
                'dispersion_pass': int(disp_pass),
                'rf_size_normal': normal_rf_size,
                'rf_size_inverted': inverted_rf_size,
                'rf_size_passes': int(rf_pass)
            }

            self.criteria_out['cell_{:03d}'.format(c)] = {
                **self.criteria_out['cell_{:03d}'.format(c)],
                **temp_dict
            }

        return self.is_IEBC
    
    def _calc_mean_resultant(self, ratemap):
        """
        Calculate the mean resultant vector, length, and angle for a rate map.

        Parameters
        ----------
        ratemap : np.ndarray
            2D array (angular x distance) of firing rates.

        Returns
        -------
        mr : complex
            Mean resultant vector (complex value).
        mean_resultant_length : float
            Length of the mean resultant vector.
        mean_resultant_angle : float
            Angle of the mean resultant vector (radians, [0, 2pi]).
        """

        N_angular_bins, N_distance_bins = ratemap.shape
        angs_rad = np.deg2rad(np.arange(0, 360, self.ray_width))

        # create a meshgrid of angles and distances
        angs_mesh, dist_mesh = np.meshgrid(angs_rad, self.dist_bin_cents, indexing='ij')

        # calculate the mean resultant vector
        mr = np.sum(ratemap * np.exp(1j * angs_mesh)) / (N_angular_bins * N_distance_bins)
        
        mean_resultant_length = np.abs(mr)
        mean_resultant_angle = np.arctan2(np.imag(mr), np.real(mr))

        if mean_resultant_angle < 0:
            mean_resultant_angle += 2 * np.pi

        return mr, mean_resultant_length, mean_resultant_angle


    def _calc_single_ratemap_subsetting(self, c, inds):
        """
        Calculate a rate map for a single cell using only a subset of frames.

        Parameters
        ----------
        c : int
            Cell index.
        inds : array-like
            Indices of frames to use.

        Returns
        -------
        rate_map : np.ndarray
            2D array (angular x distance) of firing rates for the subset.
        """
        spikes = self.data['norm_spikes'][c, inds]
        kept_indices = np.nonzero(self.useinds)[0]
        mask_in_target = np.isin(kept_indices, inds)
        ray_distances = self.ray_distances[mask_in_target, :]

        N_angular_bins = int(360 / self.ray_width)
        N_distance_bins = len(self.dist_bin_edges) - 1

        rate_map = np.zeros((N_angular_bins, N_distance_bins))

        for f in range(len(spikes)):
            for a, ang in enumerate(np.arange(0, 360, self.ray_width)):
                for d, dist_bin_start in enumerate(self.dist_bin_edges[:-1]):
                    dist_bin_end = dist_bin_start + self.dist_bin_size
                    if (ray_distances[f, a] >= dist_bin_start) and (ray_distances[f, a] < dist_bin_end):
                        rate_map[a, d] += spikes[f]

        occupancy = self.calc_occupancy(inds=inds)

        rate_map /= occupancy + 1e-6

        return rate_map


    def _calc_correlation_across_split(self, c, ncnk=20, corr_thresh=0.6):
        """
        Calculate split-half reliability for a cell's rate map.
        Splits data into chunks, shuffles, and compares two halves.
        Smooths both split maps before computing correlation.

        Parameters
        ----------
        c : int
            Cell index.
        ncnk : int, optional
            Number of chunks to split data into (default 20).
        corr_thresh : float, optional
            Correlation threshold for passing (default 0.6).

        Returns
        -------
        corr : float
            2D correlation coefficient between split maps.
        passes : bool
            Whether correlation exceeds threshold.
        """
        # Get absolute indices of frames that are used (after all masking)
        abs_inds = np.where(self.useinds)[0]
        n_used = len(abs_inds)
        
        if n_used < ncnk:
            ncnk = n_used

        cnk_sz = n_used // ncnk
        chunk_order = np.arange(ncnk)

        np.random.shuffle(chunk_order)
        
        split1_inds = []
        split2_inds = []
        for cnk in chunk_order[:(ncnk // 2)]:
            _inds = np.arange(cnk_sz * cnk, min(cnk_sz * (cnk + 1), n_used))
            split1_inds.extend(_inds)
        
        for cnk in chunk_order[(ncnk // 2):]:
            _inds = np.arange(cnk_sz * cnk, min(cnk_sz * (cnk + 1), n_used))
            split2_inds.extend(_inds)
        
        split1_inds = np.array(np.sort(split1_inds)).astype(int)
        split2_inds = np.array(np.sort(split2_inds)).astype(int)
        
        # Map split indices to absolute indices in the original data
        split1_abs = abs_inds[split1_inds[split1_inds < n_used]]
        split2_abs = abs_inds[split2_inds[split2_inds < n_used]]
        
        rm1 = self._calc_single_ratemap_subsetting(c, split1_abs)
        rm2 = self._calc_single_ratemap_subsetting(c, split2_abs)
        
        # Smooth the two split rate maps before correlation
        rm1_smooth, rm2_smooth = self.smooth_map_pair(rm1, rm2)
        
        # Calculate 2D correlations on smoothed maps
        corr = fm2p.corr2_coeff(rm1_smooth, rm2_smooth)

        # Check if the correlation exceeds the threshold
        passes = corr > corr_thresh


        temp_dict = {
            'split_rate_map_1': rm1_smooth,
            'split_rate_map_2': rm2_smooth,
        }
        self.criteria_out['cell_{:03d}'.format(c)] = {
            **self.criteria_out['cell_{:03d}'.format(c)],
            **temp_dict
        }

        return corr, passes
    
    def _test_mean_resultant_across_shuffles_mp(self, c, mrl, n_shfl=100, mrl_thresh_position=99):
        """
        Test mean resultant length (MRL) against shuffled spike trains using multiprocessing.

        Parameters
        ----------
        c : int
            Cell index.
        mrl : float
            Observed mean resultant length.
        n_shfl : int, optional
            Number of shuffles (default 100).
        mrl_thresh_position : float, optional
            Percentile for threshold (default 99).

        Returns
        -------
        use_mrl_thresh : float
            Threshold MRL from shuffled distribution.
        passes : bool
            Whether observed MRL exceeds threshold.
        """
        n_proc = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=n_proc)

        mp_param_set = [
            pool.apply_async(
                calc_shfl_mean_resultant_mp,
                args=(
                    self.data['norm_spikes'][c, :].copy(),
                    self.useinds,
                    self.occupancy,
                    self.ray_distances,
                    self.ray_width,
                    self.dist_bin_edges,
                    self.dist_bin_size,
                    self.dist_bin_cents,
                    self.is_IEBC[c]
                )
            ) for n in range(n_shfl)
        ]
        shuffled_mrls = [result.get() for result in mp_param_set]

        shuffled_mrls = np.array(shuffled_mrls)
        use_mrl_thresh = np.percentile(shuffled_mrls, mrl_thresh_position)
        passes = mrl > use_mrl_thresh

        temp_dict = {
            'shuffled_mrls': shuffled_mrls
        }

        self.criteria_out['cell_{:03d}'.format(c)] = {
            **self.criteria_out['cell_{:03d}'.format(c)],
            **temp_dict
        }

        return use_mrl_thresh, passes

    
    def _test_mean_resultant_across_shuffles(self, c, mrl, n_shfl=100, mrl_thresh_position=99, use_mp=True):
        """
        Test mean resultant length (MRL) against shuffled spike trains.
        Optionally uses multiprocessing.

        Parameters
        ----------
        c : int
            Cell index.
        mrl : float
            Observed mean resultant length.
        n_shfl : int, optional
            Number of shuffles (default 100).
        mrl_thresh_position : float, optional
            Percentile for threshold (default 99).
        use_mp : bool, optional
            Whether to use multiprocessing (default True).

        Returns
        -------
        mrl : float
            Observed mean resultant length.
        passes : bool
            Whether observed MRL exceeds threshold.
        """
        if use_mp:
            return self._test_mean_resultant_across_shuffles_mp(c, mrl, n_shfl, mrl_thresh_position)
            
        N_frames = np.sum(self.useinds)

        shuffled_mrls = []
        for shf in range(n_shfl):
            shift_amount = np.random.randint(int(0.1*N_frames), int(0.9*N_frames))
            shifted_inds = np.roll(np.arange(N_frames), shift_amount)
            shifted_ratemap = self._calc_single_ratemap_subsetting(c, shifted_inds)
            if self.is_IEBC[c]:
                shifted_ratemap = self._invert_ratemap(shifted_ratemap)
            _, shf_mrl, _ = self._calc_mean_resultant(shifted_ratemap)
            shuffled_mrls.append(shf_mrl)
        shuffled_mrls = np.array(shuffled_mrls)
        use_mrl_thresh = np.percentile(shuffled_mrls, mrl_thresh_position)
        passes = mrl > use_mrl_thresh

        return mrl, passes
    
    def identify_boundary_cells(self, n_chunks=20, n_shuffles=20, corr_thresh=0.6, mp=True):
        """
        Classify boundary cells (EBCs) using split-half reliability and MRL criteria.

        Parameters
        ----------
        n_chunks : int, optional
            Number of chunks for split-half (default 20).
        n_shuffles : int, optional
            Number of shuffles for MRL test (default 20).
        corr_thresh : float, optional
            Correlation threshold for split-half (default 0.6).
        mp : bool, optional
            Whether to use multiprocessing for shuffles (default True).

        Returns
        -------
        criteria_out : dict
            Dictionary with classification results and metrics for each cell.
        """
        N_cells = self.rate_maps.shape[0]
        self.is_EBC = np.zeros(N_cells, dtype=bool)

        for c in tqdm(range(N_cells)):
            ratemap = self.rate_maps[c, :, :]
            if self.is_IEBC[c]:
                ratemap = self._invert_ratemap(ratemap)
            
            mr, mrl, mra = self._calc_mean_resultant(ratemap)

            # correlation coefficient criteria
            corr, corr_pass = self._calc_correlation_across_split(c, ncnk=n_chunks, corr_thresh=corr_thresh)

            # mean resultant criteria
            mrl_99_pctl, mrl_pass = self._test_mean_resultant_across_shuffles(c, mrl, n_shfl=n_shuffles, use_mp=mp)

            if corr_pass and mrl_pass:
                self.is_EBC[c] = True

            temp_dict = {
                'mean_resultant': mr,
                'mean_resultant_length': mrl,
                'mean_resultant_angle': mra,
                'corr_coeff': corr,
                'corr_pass': int(corr_pass),
                'mrl_99_pctl': mrl_99_pctl,
                'mrl_pass': int(mrl_pass)
            }

            self.criteria_out['cell_{:03d}'.format(c)] = {
                **self.criteria_out['cell_{:03d}'.format(c)],
                **temp_dict
            }

        return self.criteria_out
    
    def identify_responses(self, use_angle='head', use_light=False, use_dark=False, skip_classification=False):
        """
        Full pipeline for boundary cell analysis: computes ray distances, occupancy, rate maps,
        smoothing, and classifies EBC/IEBC if requested.

        Parameters
        ----------
        use_angle : str, optional
            Which angle to use ('head', 'pupil', or 'ego'). Default is 'head'.
        use_light : bool, optional
            Restrict to light condition (default False).
        use_dark : bool, optional
            Restrict to dark condition (default False).
        skip_classification : bool, optional
            If True, skip EBC/IEBC classification (default False).

        Returns
        -------
        data_out : dict
            Dictionary with all computed maps, metrics, and classifications.
        """
        if use_light:
            assert self.data['ltdk'] == True, 'Data must be preprocessed with light conditions.'
            print('  -> Calculating boundary responses for light condition.')
            useinds = self.data['ltdk_state_vec'].copy() == 1

        elif use_dark:
            assert self.data['ltdk'] == True, 'Data must be preprocessed with dark conditions.'
            print('  -> Calculating boundary responses for dark condition.')
            useinds = self.data['ltdk_state_vec'].copy() == 0

        elif (not use_light) and (not use_dark):
            print('  -> Calculating boundary responses for all frames.')
            useinds = np.ones(self.data['norm_spikes'].shape[1])

        self.useinds = useinds

        # was shited by -2 frames (spikes shifted as: [2, 3, 4, 0, 1])
        # changed to shifted +2 frames on 8/18/25 (spikes shifted as [3, 4, 0, 1, 2])
        # last version with -2 was _v5.h5; Now testing +2 with _v6_posroll.h5
        # self.data['norm_spikes'] = np.roll(self.data['norm_spikes'], 2, axis=1)
        self.useinds = self.useinds * (self.data['speed'] > 2.)
        # calculate potential angles
        if use_angle == 'head':
            self.calc_allo_yaw()

        elif use_angle == 'pupil':
            self.calc_allo_pupil()

        elif use_angle == 'ego':
            self.calc_ego()

        # calculate all ray distances
        print('  -> Calculating ray distances.')
        _ = self.get_ray_distances(angle=use_angle)

        print('  -> Calculating occupancy.')
        self.occupancy = self.calc_occupancy(inds=self.useinds)

        print('  -> Calculating rate maps.')
        _ = self.calc_rate_maps()

        print('  -> Smoothing rate maps (just for later visualization).')
        _ = self.smooth_rate_maps()

        if not skip_classification:
            print('  -> Identifying inverse boundary cells.')
            _ = self.identify_inverse_responses()

            print('  -> Identifying boundary cells.')
            _ = self.identify_boundary_cells()

        data_out = {
            'occupancy': self.occupancy,
            'rate_maps': self.rate_maps,
            'smoothed_rate_maps': self.smoothed_rate_maps,
            'ray_width': self.ray_width,
            'max_dist': self.max_dist,
            'dist_bin_size': self.dist_bin_size,
            'bin_dist_edges': self.dist_bin_edges,
            'dist_bin_cents': self.dist_bin_cents,
            'ray_distances': self.ray_distances,
            'angle_rad': np.deg2rad(np.arange(0, 360, self.ray_width))
        }
        if not skip_classification:
            final_clas = {
                'is_IEBC': self.is_IEBC.astype(int),
                'is_EBC': self.is_EBC.astype(int)
            }
            data_out = {
                **data_out,
                **self.criteria_out,
                **final_clas
            }

        self.data_out = data_out

        return data_out

    def save_results(self, savepath):
        """
        Save results to HDF5 file using fm2p.write_h5.

        Parameters
        ----------
        savepath : str
            Path to save the HDF5 file.
        """
        data_out = convert_bools_to_ints(self.data_out)

        fm2p.write_h5(savepath, data_out)


