"""
Genreal image stack operations.

Author: DMM, 2023
last modified 2024
"""

import tifffile
import os
import cv2
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import skimage.registration
import scipy.ndimage


def norm_arr(A, min_=None, max_=None):
    """ Normalize an array between two values.

    Parameters
    ----------
    A : np.ndarray
        Array to normalize.
    min_ : int or float (optional)
        Minimum value to scale array values to. If no value
        is provided, use the array's minimum.
    max_ : int or float (optional)
        Sam as min_ for the maximum value.
    
    Returns
    -------
    _a : np.ndarray
        Array with the asme shape as input argument `A`, with
        contained values normalized between the chosen bounds.
    """

    if min_ is None:
        min_ = np.nanmin(A)
    if max_ is None:
        max_ = np.nanmax(A)

    _a = A + np.abs(min_)
    _a = _a / max_

    return _a


def register_stack_to_template(stack, template=None):
    """ Register a stack of images to a template image.

    Parameters
    ----------
    stack : np.ndarray
        Image stack as a 3D numpy array.

    Returns
    -------
    stack : np.ndarray
        Image stack with shifted images.
    extras : dict
        Dictionary of extra variables. Variables are
        'x_shift', 'y_shift', and 'shifterr', (the shift
        values and the error for each frame, respectively).
    """

    print('Registering image stack to template.')

    if template is None:
        template = stack[0,:,:].copy()

    # Initialize arrays to store shift values
    x_shift = np.zeros(np.size(stack, axis=0))
    y_shift = np.zeros(np.size(stack, axis=0))
    shifterr = np.zeros(np.size(stack, axis=0))


    # print('Starting stack registration')
    for i in tqdm(range(np.size(stack, axis=0))):
        # shift image to match template
        shift, error, _ = skimage.registration.phase_cross_correlation(
            reference_image=template,
            moving_image=stack[i,:,:],
            upsample_factor=4
        )

        x_shift[i] = shift[0]
        y_shift[i] = shift[1]
        shifterr[i] = error

        # Apply shift to image
        stack[i,:,:] = scipy.ndimage.shift(
            stack[i,:,:],
            shift,
            mode='constant',
            cval=np.nan
        )

    # Make a dictionary of extras to return
    extras = {
        'x_shift': x_shift,
        'y_shift': y_shift,
        'shifterr': shifterr
    }

    # Return stack with shifted images
    return stack, extras


def load_tif_stack(path, rotate=False, ds=1.0, doReg=False, doNorm=False):
    """ Load a tif stack into a numpy array.

    Before running this function, make sure that the tif stack is a single
    multi-page tif file. This conversion can be done with the function
    `imgtools.tif_convert()` or the matlab function `subroutine_tifConvert.m`
    from the Goard lab 2P post-processing repository.

    Parameters
    ----------
    path : str
        Path to tif stack, which needs to be a single multi-page
        tif file.
    rotate : bool
        Rotate the image by 180 deg. Default value is False.
    ds : float
        Downsample the image by this factor. If this value is set
        to 1, the image will remaind and full-size. If it is set to
        0.25, images are resized to one-quarter of the original size.
    doReg : bool
        Register the image stack to a template. Default is True.
    doNorm : bool
        Normalize the image using its minimum and maximum values as
        the upper and lower bounds.

    Returns
    -------
    tif_array : np.ndarray
        Image stack as a numpy array.
    """

    tif_array = tiff.imread(path)

    # Rotate by 180 deg
    if rotate is True:
        for i in range(np.size(tif_array, axis=0)):
            tif_array[i,:,:] = np.flipud(np.fliplr(tif_array[i,:,:]))

    # Downsample, to 1/4 original resolution along axis 1 and 2
    if ds != 1:
        tif_array = tif_array[:, ::int(1/ds), ::int(1/ds)]

    # Image stack registration
    if doReg is True:
        tif_array, _ = register_stack_to_template(tif_array)

    if doNorm is True:
        tif_array = norm_arr(tif_array)

    return tif_array


def multipart_tif_to_avi(searchpath):
    """ Read a multi-part TIF and write as an avi file.

    Parameters
    ----------
    searchpath : str
        Path in which to search for individual tifs.

    Returns
    -------
    video_savepath : str
        The savepath of the .avi written to disk.
    """

    filelist = [os.path.join(searchpath, f) for f in os.listdir(searchpath)]

    # get dims of first item
    f = filelist[0]
    imgs = load_tif_stack(f, doReg=False, doNorm=False)
    total_frames = 0
    for f in filelist:
        total_frames += np.size(load_tif_stack(f, doReg=False, doNorm=False), 0)

    print('Found {} frames.'.format(total_frames))

    imgstack = np.empty([total_frames, 512, 640, 3])

    filled_to = 0
    print('Reading tif blocks...')
    for f in tqdm(filelist):
        im = load_tif_stack(f, doReg=False, doNorm=False)
        will_add = np.size(im,0)
        imgstack[filled_to:filled_to+will_add,:,:,:] = im.copy()
        filled_to += will_add

    video_savepath = os.path.join(searchpath, 'full_video.avi')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(
        video_savepath,
        fourcc, 60.,
        (
            np.size(imgstack, 2),
            np.size(imgstack, 1)
        )
    )

    print('Writing avi file...')
    for i in tqdm(range(np.size(imgstack, 0))):

        im = imgstack[i,:,:,:]
        im = im.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        video.write(im)

    cv2.destroyAllWindows()
    video.release()

    return video_savepath


def read_tif_frame(file_path, frame_index):
    """
    Read a single frame from a TIFF stack without loading the entire file.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.
    frame_index : int
        Zero-based index of the frame to read.

    Returns
    -------
    numpy.ndarray
        The requested frame as an array.
    """
    with tifffile.TiffFile(file_path) as tif:
        num_pages = len(tif.pages)
        if frame_index < 0 or frame_index >= num_pages:
            raise IndexError(f"Frame index {frame_index} out of range (0–{num_pages-1})")
        frame = tif.pages[frame_index].asarray()
    return frame


def read_tif_until(file_path, last_frame=3600):
    """
    Read frames sequentially from a TIFF stack up to (and including) last_frame,
    without loading the entire stack beyond that point.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.
    last_frame : int, optional (default=3600)
        Zero-based index of the last frame to read.

    Returns
    -------
    numpy.ndarray
        Stack of frames up to the given frame, shape = (num_frames, height, width).
    """
    frames = []
    with tifffile.TiffFile(file_path) as tif:
        num_pages = len(tif.pages)
        stop = min(last_frame + 1, num_pages)
        for i in range(stop):
            frames.append(tif.pages[i].asarray())
    return np.stack(frames, axis=0)


