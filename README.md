# Freely moving two-photon calcium imaging

To install this as a package within a conda environment, run `pip install -e .` in the directory of this repository.

To create the conda environment from the explicit package versions in "spec-file.txt", run `conda create --name fm1 --file spec-file.txt`. New specification files can be created with `conda list --explicit > spec-file.txt`.

### Preprocess a somatic recording

**1. Analyze 2P data.**

Analyze the two-photon imaging .tif stack using Suite2p by running `python -m suite2p` in a conda environment with suite2p version 0.14.2 installed (suite2p is not in the fm1 conda environment). For the GCaMP6s sensor, use `tau=1.2`. Use the appropriate sample rate. For most recordings, this is `fs=7.49`. Go through the segmented cells and ensure that only good cells are included.

**2. Create the config file.**

Make a copy of the file "config.yaml", save it in the main directory of the recording, i.e., ".../250101_DMM_DMM000_pillar/config.yaml". Add the recording directory to the file (in the `spath` field).

**3. Preprocesss the recording.**

Activate the conda environment (`conda activate fm1`). Preprocess the recording by running `python -m fm2p.preprocess -cfg  K:/Mini2P/250101_DMM_DMM000_pillar/config.yaml` with the path to the config file create in the previous step. If you don't include the `-cfg` flag, a dialog box will open in which you can select the config file.

During this process, three dialog boxes will open showing an example frame from the top-down video.

In the first window, four blue points are placed on the corners of the arena. Once a point is placed, you cannot adjust it. They *must* be placed in the following order: top-left, top-right, bottom-left, bottom-right.

In the next window, eight red points are placed around the perimeter of the checkerboard pillar in the arena. They should be placed in continuous order, such that if the first point is at the top of the arena, the next should be in either the top-left or top-right corner and continue around the perimeter of the pillar in counter-clockwise or clockwise order, respectively. Place these points on the edge of the pillar at its highest point, ignoring that the base of the pillar may be in a different position depending on how close to the edge of the field of view the pillar was in this recording.

The third window will show an orange line tracing the points placed in the previous window. Click on the orange line and drag it so that the circle is now over the base of the pillar instead of at the top of the pillar. For a pillar in the center of the arena, these will be identical positions. For a pillar near the edge of the arena, there may be a large distance between these two positions. When you're happy with its position, close this window.

This pipeline will:
* deinterlace the eyecamera video
* run DeepLabCut to measure the edges of the animals pupil and to track the animal's position from the top-down camera
* measure the animal's position and orientation in the arena, align the behavioral measures from the eyecamera video to the 2P and top-down data using the TTL voltages read in through Bonsai
* fit an ellipse to the pupil and measure the pupil orientation via ellipse tilt
* read in Suite2p outputs, calculate dF/F, run the OASIS algorithm to infer spikes from fluorescence data, and create timestamps for each two-photon frame
* interpolate pupil orientations to two-photon and top-down timestamps
* measure the position of the arena's pillar in retinocentric and egocentric coordinates

A single .h5 file will be written in the directory of the recording with all preprocessed data.

### Preprocess an axonal recording

**1. Denoise the 2P data.**

There is noise in the tif stack which is likely from the resonance scanner. This is much worse in axonal than in somatic recordings, when the laser power is higher and the SNR is worse. A function from the `imgtools` repository [here](https://github.com/dylanmmartins/image-tools) is used to subtract the noise, which shows up as thick bands of onise (~50 pixels wide) that extend vertically to the top and bottom of the image. They sweep slowly both leftward and rightward, with changing overlap over time. TO suibtract this, run `python -m imgtools.resscan_denoise` and select the tif stack in the dialog box that opens. This code is memory intensive and needs to be run on a computer with >128 GB RAM. In addition to a readme with some details and a PDF of diagnostic figures, the code will write two tif files that should have the same

**2. Analyze the 2P data.**

Use the Goard lab two-photon calcium post processing pipeline repository [here](https://github.com/ucsb-goard-lab/Two-photon-calcium-post-processing), which I run with Matlab 2023b. Run the function `A_ProcessTimeSeries.m` without image registration. Then, run `B_DefineROI.m`, perform "Activity Map" segmentation with default values except for the "minimum pixel size" which should be changed to 5. Next, run `C_ExtractDFF.m` choosing "Local Neuropil Subtraction" and choosing "Yes" to "Weight subtraction to minimize signal-noise correlation?"

**3. Create the config file.**




to create yaml file: `conda env export --from-history > environment.yml`