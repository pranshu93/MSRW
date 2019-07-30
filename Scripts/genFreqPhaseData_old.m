window = 512;
remove = 0;
remove_begin = 2;
remove_last = remove_begin;
total_remove = remove_begin + remove_last; 
data_dir = 'val';

if strcmp(data_dir, 'train')
  data_prefix = 'bb_3class_winlen_512_winindex_all_train'; 
  outfile = '3class_winlen_512_train.csv';
elseif strcmp(data_dir, 'test')
  data_prefix = 'bb_3class_winlen_512_winindex_all_test';
  outfile = '3class_winlen_512_test.csv';
elseif strcmp(data_dir, 'val')
  data_prefix = 'bb_3class_winlen_512_winindex_all_val';
  outfile = '3class_winlen_512_val.csv';
end

dir_prefix = '/scratch/sk7898/Bumblebee/bb_3class_winlen_512_winindex_all/';
dir_human = fullfile(dir_prefix, data_prefix, 'Human');
dir_noise = fullfile(dir_prefix, data_prefix, 'Noise');
dir_nonhuman = fullfile(dir_prefix, data_prefix, 'Nonhuman');
data_dirs = {dir_noise, dir_human, dir_nonhuman};
num_classes = 3;
data = {};
data_len = {0, 0, 0};
class_count = {0, 0, 0};

for class=1:num_classes
    n = 1;
    cd(data_dirs{class});
    fullNames=dir;
    
    fprintf('Data Directory: %s \n', data_dirs{class})
    for j=1:length(fullNames)
         s=fullNames(j).name;
         k=strfind(s,'.data');
         if ~isempty(k) && k>=2 && k+4==length(s)
             data{class, n}=s(1:k-1);
             data_len{class}=data_len{class}+1;
             n = n + 1;
         end
    end   
end

cnn_data = [];

fprintf('Windowing Data..................\n')
cd('/scratch/sk7898/MATLAB_Scripts/Scripts')
for class=1:num_classes
    n = data_len{class};
    for i=1:n-1
        d = getFreqData(data{class, i}, data_dirs{class});
        cn = size(d);
        if remove == 1
            if (cn(1) - total_remove) > 0
                class_count{class} = class_count{class} + cn(1)- total_remove;
                cnn_data = vertcat(cnn_data, d(remove_begin+1:cn(1)- remove_last,:));
	    end
	else
	    class_count{class} = class_count{class} + cn(1);
            cnn_data = vertcat(cnn_data, d);
        end        
    end
    fprintf('Finished Generating %d windows for %s class\n', class_count{class}, data_dirs{class})
end

cmag = cnn_data(:,1:window);
cphi = cnn_data(:,window+1:window*2);
cmag = (cmag - (mean(mean(cmag)).*ones(size(cmag))))./1000;
cphi = (cphi - (mean(mean(cphi)).*ones(size(cphi))));
cmag = round(cmag .*100)./100;
cphi = round(cphi .*100)./100;

cnn_datapoints = 0;
for class=1:num_classes
    cnn_datapoints = cnn_datapoints + class_count{class};
end
    
labels = zeros(cnn_datapoints, 1);
for class=2:num_classes
    begin_idx = class_count{class} + 1;
    if class == num_classes
        end_idx = cnn_datapoints;
    else
        end_idx = class_count{class} + class_count{class+1};
    end
    labels(begin_idx:end_idx, 1) = class - 1;
end

train(:,1:2:(window*2)-1) = cmag;
train(:,2:2:window*2) = cphi;
data = horzcat(train, labels);

fprintf('Writing to csv\n')
cd(dir_prefix)
csvwrite(outfile,data)
