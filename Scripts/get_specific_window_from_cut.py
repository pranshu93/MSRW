import numpy as np
import os
import glob
import shutil
from helpermethods import Data2IQ

def extract_windows(indirs, outdir, class_label, stride, winlen,
                    windows_all=False, get_window_position=0, samprate=256, minlen_secs=1):
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
    outdir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Windowed'

    ###################################################################################################
    ####################################### BALANCED DATASETS #########################################
    ###################################################################################################

    print('----------------Human BumbleBee Targets (balanced)----------------')
    # Bumblebee human cuts
    extract_windows(indirs=[
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/arc_1 (Humans_Gym balls)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/bv_4 (Humans_Cars)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/ceiling_238_10 (Humans_Gym balls)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/11-30-2011/Human',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/kh_3 (Humans_Gym balls)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/prb_2 (Humans_Gym balls)/Human/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Parking garage orthogonal (Humans)/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Parking garage radial (Humans)/'
    ],
        outdir=outdir,
        class_label='Human',
        stride=384,
        winlen=384)

    print('----------------Non-human BumbleBee Targets (balanced)----------------')
    # Bumblebee dog cuts
    data_nonhumans = extract_windows(indirs=[
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/arc_1 (Humans_Gym balls)/Ball/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/bv_4 (Humans_Cars)/Car/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/ceiling_238_10 (Humans_Gym balls)/Ball/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/kh_3 (Humans_Gym balls)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/prb_2 (Humans_Gym balls)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/osu_farm_meadow_may24-28_2016_subset_113 (Cattle)/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Radar_site1_hilltop (Cattle)/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Radar_site2_creamery_subset_113 (Cattle)/'
    ],

        outdir=outdir,
        class_label='Nonhuman',
        stride=384,
        winlen=384)
    exit()

    ###################################################################################################
    ########################################### FULL DATASETS #########################################
    ###################################################################################################

    print('----------------Human BumbleBee Targets (full)----------------')
    # Bumblebee human cuts
    extract_windows(indirs=[
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/arc_1 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/bv_4 (Humans_Cars)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/ceiling_238_10 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/combined_5 (Humans_Dogs)/11-30-2011/Human',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/combined_5 (Humans_Dogs)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/kh_3 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/prb_2 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/Parking garage orthogonal (Humans)/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/Parking garage radial (Humans)/'
    ],
        outdir=outdir,
        class_label='Human',
        stride=384,
        winlen=384)

    print('----------------Non-human BumbleBee Targets (full)----------------')
    # Bumblebee dog cuts
    data_nonhumans = extract_windows(indirs=[
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/arc_1 (Humans_Gym balls)/Ball/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/bv_4 (Humans_Cars)/Car/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/ceiling_238_10 (Humans_Gym balls)/Ball/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/combined_5 (Humans_Dogs)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/kh_3 (Humans_Gym balls)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/prb_2 (Humans_Gym balls)/Dog/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/osu_farm_meadow_may24-28_2016 (Cattle)/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/Radar_site1_hilltop (Cattle)/',
        '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets_full/Radar_site2_creamery (Cattle)/'
    ],

        outdir=outdir,
        class_label='Nonhuman',
        stride=384,
        winlen=384)