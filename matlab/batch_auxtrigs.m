 list=dir('**/*.tif*');
 
 for idx = 1 : length(list)
     
     try
     imf = scanimage.util.opentif([list(idx).folder, '\', list(idx).name]);
     stamp=[];
     k=1;
     frameSec = imf.frameTimestamps_sec;
    for i = 1:length(imf.auxTrigger0)
        tmp = imf.auxTrigger0{i};
        if ~isempty(tmp)
            i;
            tmp;
            hold on, plot(i, tmp, '.');
            stamp(k)=tmp(1);
            k=k+1;
        end
    end
    cd(list(idx).folder)
    filename = string([list(idx).name(1:end-4), '_imf']);
    save(filename,'stamp','frameSec')
     end
 end