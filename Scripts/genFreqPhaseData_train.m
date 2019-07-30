remove = 0;
remove_begin = 2;
remove_last = remove_begin;
total_remove = remove_begin + remove_last; 
data_dir = 'train';
classes = {'Human', 'Bike'};

dir_prefixes = {
	        '/scratch/sk7898/austere/classification_data_windowed/winlen_384_winindex_all/pedbike_class_winlen_384_winindex_all/',
	        '/scratch/sk7898/austere/classification_data_windowed/winlen_512_winindex_all/pedbike_class_winlen_512_winindex_all/',
               };

prefixes = {'pedbike_class_winlen_384_winindex_all_',
	    'pedbike_class_winlen_512_winindex_all_',
            };

windows = {384, 512};

for dataset=1:length(dir_prefixes)	      
  if strcmp(data_dir, 'train')
    data_prefix = strcat(prefixes{dataset}, 'train'); 
    outfile = strcat(data_prefix, '_freq.csv');
  elseif strcmp(data_dir, 'test')
    data_prefix = strcat(prefixes{dataset}, 'test');
    outfile = strcat(data_prefix, '_freq.csv');
  elseif strcmp(data_dir, 'val')
    data_prefix = strcat(prefixes{dataset}, 'val');
    outfile = strcat(data_prefix, '_freq.csv');
  end

  data = {};
  data_len = {};
  data_dirs = {};
  class_count = {};
  window = windows{dataset};

  for class=1:length(classes)
    data_len{class}=0;
    class_count{class}=0;
    data_dirs{class}=fullfile(dir_prefixes{dataset}, data_prefix, classes{class});
  end

  for class=1:length(classes)
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
  for class=1:length(classes)
      n = data_len{class};
      for i=1:n-1
	  d = getFreqData(data{class, i}, data_dirs{class}, window);
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
  for class=1:length(classes)
      cnn_datapoints = cnn_datapoints + class_count{class};
  end

  labels = zeros(cnn_datapoints, 1);
  for class=2:length(classes)
      begin_idx = class_count{class} + 1;
  if class == length(classes)
	  end_idx = cnn_datapoints;
      else
	  end_idx = class_count{class} + class_count{class+1};
      end
      labels(begin_idx:end_idx, 1) = class - 1;
  end

  fprintf('Concatenating data and labels.................\n')
  train = zeros(cnn_datapoints, window*2);
  train(:,1:2:(window*2)-1) = cmag;
  train(:,2:2:window*2) = cphi;
  data = horzcat(train, labels);

  fprintf('Writing to csv\n')
  cd(dir_prefixes{dataset})
  csvwrite(outfile,data)
end
