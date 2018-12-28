function train_row = getdata(fileName, path_dir)

addpath('C:\Users\Sangeeta\Desktop\MATLAB_Scripts\Scripts')
cd(path_dir);
[I,Q,N]=Data2IQ(ReadBin([fileName,'.data']));

%dcI = 2044;   % enable when do test on dummy data
%dcQ = 2048;
dcI = median(I); %median or mean
dcQ = median(Q);
Data = (I-dcI) + 1i*(Q-dcQ);

Rate=250;
FftWindow = Rate;
Nfft = Rate;
FftStep = round(1/4*FftWindow);

%DeltaF = Rate/Nfft;
%RelFreq = Wrap([0 : Nfft-1], -Nfft/2, Nfft/2);
%Freq = DeltaF * RelFreq;

%s = spectrogram(x,window,noverlap,nfft)
S = spectrogram(Data, FftWindow, FftWindow - FftStep, Nfft);
A = abs(S);
phi = angle(S);
A = A';
phi = phi';
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
