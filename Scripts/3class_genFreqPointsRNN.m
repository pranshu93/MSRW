function genFreqPointsRNN(prefix, basepath, folder_noise, folder_neg, folder_pos, outpath, cut_length_windows)
%num = 1000;

    function train_row = getFreqData(fileName, path_dir)
        cd(path_dir);
        [I,Q,N]=Data2IQ(ReadBin([fileName,'.data']));
        
        %dcI = 2044;   % enable when do test on dummy data
        %dcQ = 2048;
        dcI = median(I); %median or mean
        dcQ = median(Q);
        Data = (I-dcI) + 1i*(Q-dcQ);
        
        %Rate=256;
        Rate=window;
        FftWindow = Rate;
        Nfft = Rate;
        FftStep = round(1/4*FftWindow);
        
        %DeltaF = Rate/Nfft;
        %RelFreq = Wrap([0 : Nfft-1], -Nfft/2, Nfft/2);
        %Freq = DeltaF * RelFreq;
        
        %s = spectrogram(x,window,noverlap,nfft)
        S = spectrogram(Data, FftWindow, FftWindow - FftStep, Nfft);
        S = reshape(S, 1, []);
        A = abs(S);
        phi = angle(S);
        %A = A';
        %phi = phi';
        train_row = horzcat(A,phi);
        
        %TimeFreq = spectrogram_nohamming(Data, FftWindow, FftWindow - FftStep, Nfft, Rate);
        %numWindows = size(TimeFreq,2);
        
        %Out = zeros(numWindows,FftWindow);
        %thr_sqr_matlab_log = 14.44;
        %14.14->10, 14.14 is chosen in matlab, 10 in c#
        %thr_sqr_matlab = 10^(thr_sqr_matlab_log/10)*25238; % because S^2/P = 25238
        %x = TimeFreq';
        
        %for j = 1:FftWindow
        %Out(:,j) =  abs(x(:,j)).^2 > thr_sqr_matlab;
        %Out(:,j) =  abs(x(:,j)) > thr_sqr_matlab^0.5;
        %end
    end

window = 256;
remove = 0;
remove_begin = 0;
remove_last = remove_begin;
total_remove = remove_begin + remove_last;

data_neg = fullfile(basepath,folder_neg); %'C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Data_Repository\Bike_Human\bikes_first_window_start'; % num2str(num)];
data_pos = fullfile(basepath,folder_pos); %'C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Data_Repository\Bike_Human\humans_first_window_start'; % num2str(num)];
train_data = [];

cd(data_neg);
negFullNames=dir;
negFiles={};  % first 2 file is '.' and '..'
n = 1;

for j=1:length(negFullNames)
    s=negFullNames(j).name;
    k=strfind(s,'.data');
    if ~isempty(k) && k>=2 && k+4==length(s)
        negFiles{n}=s(1:k-1);
        n=n+1;
    end
end

cd(data_pos);
posFullNames=dir;
posFiles={};  % first 2 file is '.' and '..'
p = 1;

for j=1:length(posFullNames)
    s=posFullNames(j).name;
    k=strfind(s,'.data');
    if ~isempty(k) && k>=2 && k+4==length(s)
        posFiles{p}=s(1:k-1);
        p=p+1;
    end
end

countpos = 0;
countneg = 0;

%cd('C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Scripts')
for i=1:p-1
    d = getFreqData(posFiles{i}, data_pos);
    cp = size(d);
    if remove == 1
        if (cp(1) - total_remove) > 0
            countpos = countpos + cp(1)- total_remove;
            train_data = vertcat(train_data,d(remove_begin+1:cp(1)-remove_last,:));
        end
    else
        countpos = countpos + cp(1);
        train_data = vertcat(train_data,d);
    end
end

%cd('C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Scripts')
for i=1:n-1
    d = getFreqData(negFiles{i}, data_neg);
    cn = size(d);
    if remove == 1
        if (cn(1) - total_remove) > 0
            countneg = countneg + cn(1)- total_remove;
            train_data = vertcat(train_data,d(remove_begin+1:cn(1)- remove_last,:));
        end
    else
        countneg = countneg + cn(1);
        train_data = vertcat(train_data,d);
    end
end

cmag = train_data(:,1:cut_length_windows*window);
cphi = train_data(:,cut_length_windows*window+1:cut_length_windows*window*2);
%cmag = (cmag - (mean(mean(cmag)).*ones(size(cmag))))./1000;
%cphi = (cphi - (mean(mean(cphi)).*ones(size(cphi))));
%cmag = round(cmag .*100)./100;
%cphi = round(cphi .*100)./100;

train_labels = ones(countpos+countneg,1);
train_labels(countpos+1:countpos+countneg,1) = 0;

train(:,1:2:(cut_length_windows*window*2)-1) = cmag;
train(:,2:2:cut_length_windows*window*2) = cphi;
data = horzcat(train,train_labels);

%cd('C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Data_Repository')
cd(outpath);
%csvwrite('bike_human_full.csv',data)
csvwrite([prefix,'_RNNspectrogram.csv'],data)
end
