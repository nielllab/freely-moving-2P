

[tif_file,tif_location] = uigetfile('*.tif', 'Choose tif stack.');
[mat_file,mat_location] = uigetfile('*.mat', 'Choose Suite2P or Goard-method mat file.');

fps = 7.59;
gaussfiltSize = 30; % in seconds
maxminfiltSize = 120; % in seconds

% Get tif metadata from scanimage
fprintf("Reading image metadata.\n");
info = imfinfo(fullfile(tif_location, tif_file));
n_frames = size(info, 1);
timestamps = zeros(n_frames, 1);
for f = 1:n_frames
    [tmp, strend] = strsplit(info(f).ImageDescription, 'frameTimestamps_sec = ');
    val = strsplit(char(tmp(2)), '\nacqTriggerTimestamps_sec =');
    val = str2double(char(val(1)));
    timestamps(f) = val;
end

% Create perfect bins (real frame times have jitter)
endT = timestamps(end);
resolution = 1/fps;
bins = 0:resolution:endT;

% Suite2P output
load(fullfile(mat_location, mat_file))
n_cells = size(F,1);

fprintf("Preprocessing fluorescence.\n");

% subtract neuropil
Fcorr = F - 0.7 * Fneu;

w = gausswin(gaussfiltSize);
w = w/sum(w); % set up window for gauss filter
for i = 1:n_cells
    Fbase = filter(w,1,Fcorr(i,:)); % gauss filter
    Fbase = movmin(Fbase, maxminfiltSize*fps, 2);
    Fbase = movmax(Fbase, maxminfiltSize*fps, 2);
    Fcorr(i,:) = Fcorr(i,:) - Fbase; %f inal baseline-corrected trace
end

params.Nsamples = n_frames;
params.B = 100;
params.p = 1; 
params.f = fps;
cells = 1:n_cells;

fprintf("Finding spike times (slow).\n")

spiketimes = [];
spikerates = zeros(n_cells, length(bins)-1);
MCMCcorrs = zeros(n_cells, 1);

parfor i = cells
    % load trace into temp variable, foopsi needs double
    tracein = rescale(cast(Fcorr(i,:),'double'));

    try
        % MCMC, output is spike times in frames, not a binned spike rate
        spesti = cont_ca_sampler(tracein, params);
    catch
        fprintf('Error on cell %i, skipping this one.\n',i);
        continue
    end

    for j = 1:params.Nsamples
        % Spikes now in seconds and properly interpolated between real frame times
        spesti.ss{j,1} = interp1(1:length(timestamps), timestamps, spesti.ss{j,1});
    end
    
    % bin spikes from all samples separately
    binwidth = bins(2)-bins(1);
    spkrate = zeros(params.Nsamples, length(bins)-1);
    for j=1:params.Nsamples
        [spkrate(j,:), ~] = histcounts(spesti.ss{j,1}, bins);
        spkrate(j,:) = spkrate(j,:)./binwidth; % now spks/second
    end
    
    % Calculate mean corr b/t 50 random spike estimate samples as QC metric
    samplesize = 50;
    samps = randsample(params.Nsamples, samplesize);
    testrates = spkrate(samps,:);
    corrs = corrcoef(testrates');
    corrs = corrs.';
    meanies = tril(true(size(corrs)),-1);
    meanies_v = corrs(meanies);
    % mean correlation between 50 MCMC samples for this neuron
    meancorr = mean(meanies_v, 'omitnan');

    %save stuff
    samplekeep = randi(params.Nsamples); % only keep one MCMC sample and assoc spike rate trace
    spiketimes(i).spks = spesti.ss{samplekeep,1}; % spike times
    spikerates(i,:) = spkrate(samplekeep,:); % spk rates saved in one big mtx for all cells
    MCMCcorrs(i) = meancorr; % mean corr for QC

end

save("spikes.mat", "MCMCcorrs", "spikerates", "spiketimes", "timestamps", ...
    "Fcorr", "spks", "F", "Fneu", "iscell")
