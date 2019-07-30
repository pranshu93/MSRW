import numpy as np
import os
import glob
import shutil
from helpermethods import Data2IQ

def extract_windows(indirs, outdir, class_label, stride, winlen,
                    windows_all=True, get_window_position=0, samprate=256, minlen_secs=1):
    """
    Ref: https://github.com/dhruboroy29/MATLAB_Scripts/blob/neel/Scripts/extract_target_windows.m
    Extract sliding windows out of input data
    :param samprate: sampling rate
    :param indirs: input directory list of data
    :param outdir: output directory of windowed data
    :param class_label: data label (Target, Noise, etc.)
    :param stride: stride length in samples
    :param winlen: window length in samples
    :param minlen_secs: minimum cut length in seconds
    """

    assert isinstance(indirs, (str, list))
    assert isinstance(outdir, str)
    assert stride==winlen

    # If single directory given, create list
    if isinstance(indirs, str):
        indirs = [indirs]

    # Path to save walk length array as .csv
    #walk_length_stats_savepath = os.path.join(outdir,'3class_walk_length_stats.csv')

    # Make output directory
    if windows_all:
        suffix = 'all'
    else:
        suffix = str(get_window_position)

    outdir = os.path.join(outdir, 'winlen_' + str(winlen) + '_winindex_' + suffix, class_label)

    # Silently delete directory if it exists
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    # Silently create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Initialize cut length list to print statistics
    walk_length_stats = []
    for indir in indirs:
        assert isinstance(indir, str)

        # Find data files
        list_files = glob.glob(os.path.join(indir,'*.data'))

        for cur_file in list_files:
            # Get filename without extension
            cur_file_name = os.path.basename(os.path.splitext(cur_file)[0])

            # Read IQ samples
            I,Q,L = Data2IQ(cur_file)
            cur_walk_secs = L / samprate

            # Print data column-wise
            # print(*comp.tolist(),sep='\n')

            # Pad very short walks
            if cur_walk_secs < minlen_secs:
                continue

            # Append current walk length to stats array
            walk_length_stats.append(cur_walk_secs)

            # Extract windows
            num_windows = list(range(0, L - winlen + 1, stride))

            # If not all windows
            if not(windows_all):
                # Account for negative window positions (last, second last, etc.)
                if get_window_position < 0:
                    get_window_position_samples = (len(num_windows) + get_window_position) * stride
                else:
                    get_window_position_samples = get_window_position * stride

            for k1 in num_windows:
                if not(windows_all) and k1!=get_window_position_samples:
                    continue

                temp_I = I[k1:k1 + winlen]
                temp_Q = Q[k1:k1 + winlen]

                # uint16 cut file array
                Data_cut = np.zeros(2 * winlen, dtype=np.uint16)
                Data_cut[::2] = temp_I
                Data_cut[1::2] = temp_Q

                # Print cut column-wise
                # print(*Data_cut.astype(int), sep='\n')

                # Output filenames follow MATLAB array indexing convention
                uniqueoutfilename = os.path.join(outdir,
                                           cur_file_name + '_' + str(k1 + 1) + '_to_' + str(k1 + winlen))

                # Save to output file
                outfilename = uniqueoutfilename  + '.data'
                uniq = 1
                while os.path.exists(outfilename):
                    outfilename = uniqueoutfilename + ' (' + str(uniq) + ').data'
                    uniq += 1

                Data_cut.tofile(outfilename)

                if not(windows_all):
                    break

    # Print walk list to csv file (for CDF computation, etc)
    #with open(walk_length_stats_savepath, 'a', newline='') as myfile:
    #    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    #    wr.writerow(walk_length_stats)

    # Print walk statistics
    print('Number of cuts: ', len(walk_length_stats))
    print('Min cut length (s): ', min(walk_length_stats))
    print('Max cut length (s): ', max(walk_length_stats))
    print('Avg cut length (s): ', np.mean(walk_length_stats))
    print('Median cut length (s): ', np.median(walk_length_stats))
    print('All done!')


# Test
if __name__=='__main__':
    outdir = '/scratch/dr2915/austere/classification_data_windowed'

    for w in [384, 512]:
        print('\tProcessing winlen ', w/256, ' secs')
        print('----------------Human Austere Targets----------------')
        # Bumblebee human cuts
        extract_windows(indirs=[
            '/scratch/dr2915/austere/final_human_full_cuts'
        ],
            outdir=outdir,
            class_label='Human',
            stride=w,
            winlen=w)

        print('----------------Bike Austere Targets (radial)----------------')
        # Bumblebee dog cuts
        data_nonhumans = extract_windows(indirs=[
            '/scratch/dr2915/austere/final_bike_radial_full_cuts'
        ],

            outdir=outdir,
            class_label='Bike',
            stride=w,
            winlen=w)