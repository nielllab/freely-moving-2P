function [path1_phys] = inferSpikes(path1_phys,fps,timestamps,bins)

%{
inferSpikes Performs spike inference on fluorescence traces contained in the suite2p output container.
Requires preprocessing to baseline-corrected fluorescence traces.

inputs
path1_phys: data container. must contain Fcorr field from F_preprocess().
fps: imaging fps
timestamps: frame timestamps in seconds.
bins: bins for spike rate calculation. in seconds. 

outputs
path1_phys with updated fields:
spiketimes: nxs spike timestamps per neuron, in seconds after t=0.
spikerates: nxb binned spike rates per neuron according to bins input.
MCMCcorrs: nx1 avg correlation between 50 random spike MCMC samples, per
neuron. used as QC metric.

%}
params.Nsamples = 400;
params.B = 100;
params.p = 1; 
params.f = fps;
%cells = [1:10]; 
cells = 1:path1_phys.numcells;

path1_phys.spiketimes = [];
path1_phys.spikerates = zeros(path1_phys.numcells,length(bins)-1);
path1_phys.MCMCcorrs = zeros(path1_phys.numcells,1);
figure(); 

for i=cells
    barh(i); xlim([0 length(cells)]); yticks([]); drawnow;
    tracein = rescale(cast(path1_phys.Fcorr(i,:),'double')); %load trace into temp variable, foopsi needs double
    try
        spesti = cont_ca_sampler(tracein, params);    %% MCMC, output is spike times in frames. 
    catch
        fprintf('Error on cell %i, skipping this one.\n',i);
        continue
    end
    for j=1:params.Nsamples
        spesti.ss{j,1} = interp1(1:length(timestamps),timestamps,spesti.ss{j,1}); %spikes now in seconds and properly interpolated between real frame times.
    end
    
    %bin spikes from all samples separately
    binwidth = bins(2)-bins(1);
    spkrate = zeros(params.Nsamples,length(bins)-1);
    for j=1:params.Nsamples
        [spkrate(j,:),~] = histcounts(spesti.ss{j,1},bins);
        spkrate(j,:) = spkrate(j,:)./binwidth; %now spks/second
    end
    
    %calculate mean corr b/t 50 random spike estimate samples as QC metric
    samplesize = 50;
    samps = randsample(params.Nsamples,samplesize);
    testrates = spkrate(samps,:);
    corrs = corrcoef(testrates');
    corrs = corrs.';
    meanies = tril(true(size(corrs)),-1);
    meanies_v = corrs(meanies);
    meancorr = mean(meanies_v, 'omitnan'); %mean correlation between 50 MCMC samples for this neuron

    %save stuff
    samplekeep = randi(params.Nsamples); %only keep one MCMC sample and assoc spike rate trace
    path1_phys.spiketimes(i).spks = spesti.ss{samplekeep,1}; %spike times
    path1_phys.spikerates(i,:) = spkrate(samplekeep,:); %spk rates saved in one big mtx for all cells
    path1_phys.MCMCcorrs(i) = meancorr; %mean corr for QC

end
end

