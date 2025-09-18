function path1_phys = F_Preprocess(path1_phys,fps)
%{
F_Preprocess Calculates baseline-corrected calcium fluorescence traces
using suite2p method and taking suite2p data container as input.

input
path1_phys: suite2p data container struct
fps = imaging fps

output
path1_phys.Fcorr- baseline-corrected fluorescence
REMOVES RAW FLUORESCENCE AND NEUROPIL TRACES TO SAVE SPACE

%}

if(isfield(path1_phys,'Fcorr'))
    'Fluorescence data already preprocessed! Skipping this step...'
    return
end

[nNeurons,tPoints] = size(path1_phys.F);

path1_phys.Fcorr = path1_phys.F - 0.7*path1_phys.Fneu; %subtract neuropil
Fcorr = path1_phys.Fcorr;
gaussfiltSize = 30; %in seconds.
maxminfiltSize = 120; %in seconds.
w = gausswin(gaussfiltSize); w=w/sum(w); %set up window for gauss filter
for i=1:nNeurons %neurons
    Fbaseline = filter(w,1,Fcorr(i,:)); %gauss filter
    finalBaseline= movmin(Fbaseline,maxminfiltSize*fps,2); %then min filter 
    finalBaseline= movmax(finalBaseline,maxminfiltSize*fps,2); %then max filter 
    Fcorr(i,:) = Fcorr(i,:)-finalBaseline; %final baseline-corrected trace
end
path1_phys.Fcorr = Fcorr;
path1_phys.numcells = nNeurons;
% path1_phys.F = [];
% path1_phys.Fneu = [];
save('path1.mat','path1_phys')
end

