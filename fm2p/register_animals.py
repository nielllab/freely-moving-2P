


import os
import numpy as np
import argparse
import math
import tkinter as tk
from tkinter import Button, messagebox
from PIL import Image, ImageTk
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import fm2p


def array_to_pil(arr):
    """Convert array to PIL image, handling various dtypes and dimensions."""

    if isinstance(arr, Image.Image):
        return arr

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        # norm to 0-255
        try:
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
        except Exception:
            amin, amax = 0.0, 1.0
        if amax == amin:
            a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)

    if a.ndim == 2:
        return Image.fromarray(a, mode='L')
    if a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray(a, mode='RGB')
    if a.ndim == 3 and a.shape[2] == 4:
        return Image.fromarray(a, mode='RGBA')
    return Image.fromarray(a)


def _wheel_step_from_event(event, scale=2.0):
    """Normalize a mouse wheel event to a rotation step in degrees.
    
    Handles Windows (<MouseWheel> with event.delta), X11 (<Button-4/5> with event.num),
    and systems that send small fractional deltas (e.g. smooth scrolling).
    Returns a signed float (positive = rotate, negative = rotate opposite).
    """
    delta = getattr(event, 'delta', None)
    if delta is not None and delta != 0:
        try:
            d = float(delta)
            if abs(d) >= 1.0:
                return math.copysign(scale, d)
            return math.copysign(scale, d)
        except Exception:
            return 0.0

    num = getattr(event, 'num', None)
    if num == 4:
        return float(scale)
    if num == 5:
        return float(-scale)

    return 0.0


class AlignmentWindow:
    """Simple GUI for aligning a novel recording overlay to a reference image."""
    
    def __init__(self, ref_img_arr, novel_img_arr, ref_overlay_arr=None):
        """
        ref_img_arr: numpy array of reference image
        novel_img_arr: numpy array of novel/overlay image
        ref_overlay_arr: optional reference overlay to show on top of ref image
        """
        self.ref_img_arr = ref_img_arr
        self.novel_img_arr = novel_img_arr
        self.ref_overlay_arr = ref_overlay_arr
        
        # PIL conversions
        self.ref_img_pil = array_to_pil(ref_img_arr).convert('RGB')
        self.novel_img_pil = array_to_pil(novel_img_arr).convert('RGBA')
        
        # Convert reference overlay to RGBA if provided
        if ref_overlay_arr is not None:
            self.ref_overlay_pil = array_to_pil(ref_overlay_arr).convert('RGBA')
        else:
            self.ref_overlay_pil = None
        
        # Alignment state
        self.current_angle = 0.0
        self.current_offset = np.array([self.ref_img_pil.width // 2, 
                                        self.ref_img_pil.height // 2], dtype=float)
        self.flipped = False
        self.alpha_value = 0.7  # Alpha for novel overlay
        
        # Transform to return
        self.transform = None
        
    def _create_composite_image(self):
        """Create composite of reference + novel with current transform applied."""
        # Start with reference image
        composite = self.ref_img_pil.convert('RGBA')
        
        # Add reference overlay if available
        if self.ref_overlay_pil is not None:
            composite.paste(self.ref_overlay_pil, (0, 0), self.ref_overlay_pil)
        
        # Apply transforms to novel image
        novel = self.novel_img_pil.copy()
        
        # Apply flip if needed
        if self.flipped:
            novel = novel.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply rotation
        novel = novel.rotate(self.current_angle, expand=True)
        
        # Apply alpha
        if novel.mode == 'RGBA':
            alpha = novel.split()[3]
            alpha = Image.new('L', alpha.size, int(255 * self.alpha_value))
            novel.putalpha(alpha)
        
        # Paste novel image centered at current_offset
        paste_x = int(self.current_offset[0] - novel.width // 2)
        paste_y = int(self.current_offset[1] - novel.height // 2)
        composite.paste(novel, (paste_x, paste_y), novel)
        
        return composite
    
    def run(self):
        """Show the alignment window and return the transform (x, y, angle, flipped)."""
        root = tk.Tk()
        root.title("Align Novel Recording to Reference")
        
        # Create canvas
        canvas = tk.Canvas(root, width=self.ref_img_pil.width, 
                          height=self.ref_img_pil.height)
        canvas.pack()
        
        # Display initial composite
        def update_display():
            composite = self._create_composite_image()
            self.current_tk = ImageTk.PhotoImage(composite)
            if hasattr(update_display, 'img_id'):
                canvas.delete(update_display.img_id)
            update_display.img_id = canvas.create_image(0, 0, anchor='nw', 
                                                         image=self.current_tk)
            canvas.image = self.current_tk  # Keep reference
        
        # Mouse drag state
        drag_start = {'pos': None}
        
        def on_button_press(event):
            drag_start['pos'] = np.array([event.x, event.y], dtype=float)
        
        def on_drag_motion(event):
            if drag_start['pos'] is None:
                drag_start['pos'] = np.array([event.x, event.y], dtype=float)
            
            current = np.array([event.x, event.y], dtype=float)
            delta = current - drag_start['pos']
            self.current_offset += delta
            drag_start['pos'] = current
            update_display()
        
        def on_mouse_wheel(event):
            step = _wheel_step_from_event(event, scale=2.0)
            if step != 0.0:
                self.current_angle = (self.current_angle + step) % 360.0
                update_display()
        
        def toggle_flip():
            self.flipped = not self.flipped
            btn_flip.config(relief='sunken' if self.flipped else 'raised')
            update_display()
        
        def accept_alignment():
            self.transform = (
                float(self.current_offset[0]),
                float(self.current_offset[1]),
                float(self.current_angle),
                bool(self.flipped),
            )
            root.destroy()
        
        # Bindings
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_drag_motion)
        canvas.bind("<MouseWheel>", on_mouse_wheel)
        canvas.bind("<Button-4>", on_mouse_wheel)
        canvas.bind("<Button-5>", on_mouse_wheel)
        
        try:
            root.bind_all("<MouseWheel>", on_mouse_wheel)
            root.bind_all("<Button-4>", on_mouse_wheel)
            root.bind_all("<Button-5>", on_mouse_wheel)
        except Exception:
            pass
        
        # Control buttons
        btn_flip = Button(root, text="Flip", command=toggle_flip)
        btn_flip.pack(side='left', padx=5, pady=5)
        
        btn_accept = Button(root, text="Accept Alignment", command=accept_alignment)
        btn_accept.pack(side='left', padx=5, pady=5)
        
        btn_quit = Button(root, text="Quit", command=root.destroy)
        btn_quit.pack(side='left', padx=5, pady=5)
        
        # Display initial composite
        update_display()
        
        root.mainloop()
        
        return self.transform


def create_shared_ref():

    # choose a .mat and png directory of best widefield map
    wf_dir = fm2p.select_directory(
        'Select widefield output directory (should include mat files and reference tiff).'
    )
    ref_tif_path = fm2p.find('*.tif', wf_dir, MR=True)
    am_mat_path = os.path.join(wf_dir, 'additional_maps.mat')
    vfs_mat_path = os.path.join(wf_dir, 'VFS_maps.mat')

    im = Image.open(ref_tif_path)
    img = np.array(im)

    matfile = loadmat(vfs_mat_path)

    matfile2 = loadmat(am_mat_path)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    smlimg = 1-(smlimg-np.min(smlimg)) / 65535

    overlay = matfile['VFS_raw'].copy()

    h_over = matfile2['maps']['HorizontalRetinotopy'].copy()[0][0]
    v_over = matfile2['maps']['VerticalRetinotopy'].copy()[0][0]

    t2b = np.zeros([256, 4])
    t2b[:,3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    overlay_img = gaussian_filter(overlay,1.5)

    # save out the ref img, overlay
    out = {
        'overlay': overlay_img,
        'refimg': smlimg
    }

    savepath = os.path.join(wf_dir, 'animal_reference_{}.h5'.format(fm2p.fmt_now(c=True)))
    fm2p.write_h5(savepath, out)

def align_novel_rec_to_ref():
    """Align a novel recording to a shared reference and transform cell coordinates.
    
    This function:
    1. Loads a reference recording with its reference image and overlay
    2. Loads a novel recording's reference image
    3. Allows user to manually align the novel overlay to the reference image
    4. Applies the global-to-global transformation to all cell coordinates
    5. Saves the transformed data to a new HDF file with timestamp
    """
    
    # Load the reference composite file with global coordinates
    ref_composite_file = fm2p.select_file(
        'Select reference composite file (with global coordinates).',
        filetypes=[('HDF', '.h5'),]
    )
    ref_data = fm2p.read_h5(ref_composite_file)
    
    # Extract reference image and overlay
    if 'refimg' not in ref_data:
        raise ValueError("Reference file must contain 'refimg' key")
    ref_img = ref_data['refimg']
    ref_overlay = ref_data.get('overlay', None)
    
    # Load the novel recording's reference image
    novel_composite_file = fm2p.select_file(
        'Select novel recording composite file (to be aligned).',
        filetypes=[('HDF', '.h5'),]
    )
    novel_data = fm2p.read_h5(novel_composite_file)
    
    if 'refimg' not in novel_data:
        raise ValueError("Novel file must contain 'refimg' key")
    novel_img = novel_data['refimg']
    
    # Create alignment window - show reference image with its overlay,
    # and allow user to align the novel image on top
    aligner = AlignmentWindow(ref_img, novel_img, ref_overlay_arr=ref_overlay)
    transform = aligner.run()
    
    if transform is None:
        print("Alignment cancelled.")
        return
    
    novel_x, novel_y, novel_angle, novel_flipped = transform
    theta = math.radians(novel_angle)
    
    # Extract cell coordinates from novel data
    # Expected structure: keys are position IDs, values are arrays with cell positions
    all_transformed_positions = {}
    
    for pos_key, cell_array in novel_data.items():
        if pos_key in ('refimg', 'overlay'):
            # Skip image data, only process coordinate arrays
            continue
        
        if not isinstance(cell_array, np.ndarray) or cell_array.size == 0:
            continue
        
        # cell_array shape: [n_cells, n_coords] where coords might be [x_local, y_local, ...]
        # We'll transform each cell's x, y coordinates
        transformed_cells = cell_array.copy()
        
        if cell_array.shape[1] >= 2:
            # Apply local-to-global transformation following logic from register_tiled_locations.py
            # Assumes first two columns are x_local, y_local
            h_novel, w_novel = novel_img.shape[:2]
            
            for cell_idx in range(cell_array.shape[0]):
                x_local = float(cell_array[cell_idx, 0])
                y_local = float(cell_array[cell_idx, 1])
                
                # Scale by reference scale if applicable
                # (in this case we assume 1:1 scale since both are composite images)
                scale_factor = 1.0
                
                x_s = x_local * scale_factor
                y_s = y_local * scale_factor
                
                # Center the image
                cx = (w_novel * scale_factor) / 2.0
                cy = (h_novel * scale_factor) / 2.0
                dx = x_s - cx
                dy = y_s - cy
                
                # Apply flip if needed
                if novel_flipped:
                    dx = -dx
                
                # Apply rotation
                xr = dx * math.cos(theta) - dy * math.sin(theta)
                yr = dx * math.sin(theta) + dy * math.cos(theta)
                
                # Translate to global position (in reference frame)
                x_global = novel_x + xr
                y_global = novel_y + yr
                
                transformed_cells[cell_idx, 0] = x_global
                transformed_cells[cell_idx, 1] = y_global
        
        all_transformed_positions[pos_key] = transformed_cells
    
    # Also include reference data so we have all recordings in one file
    for pos_key, cell_array in ref_data.items():
        if pos_key not in all_transformed_positions and pos_key not in ('refimg', 'overlay'):
            all_transformed_positions[pos_key] = cell_array
    
    # Save output with timestamp to avoid overwriting
    output_dir = os.path.dirname(novel_composite_file)
    timestamp = fm2p.fmt_now(c=True)
    output_filename = f"aligned_composite_{timestamp}.h5"
    output_path = os.path.join(output_dir, output_filename)
    
    fm2p.write_h5(output_path, all_transformed_positions)
    print(f"Alignment saved to: {output_path}")
    
    # Print transform for reference
    print(f"Applied transform - x: {novel_x:.1f}, y: {novel_y:.1f}, "
          f"angle: {novel_angle:.1f}°, flipped: {novel_flipped}")


if __name__ == '__main__':

    align_novel_rec_to_ref()

