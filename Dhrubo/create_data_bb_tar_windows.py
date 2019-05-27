import sys, os
from importlib import reload

sys.path.append('../Scripts')

from Scripts.get_specific_window_from_cut import extract_windows

reload_flag = False

# Init args
init_argv = sys.argv

######################### ONLY MODIFY THESE VALUES #########################

# Type
prefix = 'bb_tar'

outdir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Windowed'

foldoutdir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/FastGRNN/Data/Bumblebee'

classdirs = ['Human', 'Nonhuman']

human_dirs = [
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/arc_1 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/bv_4 (Humans_Cars)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/ceiling_238_10 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/11-30-2011/Human',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/kh_3 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/prb_2 (Humans_Gym balls)/Human/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Parking garage orthogonal (Humans)/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Parking garage radial (Humans)/'
]

nonhuman_dirs = [
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/arc_1 (Humans_Gym balls)/Ball/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/bv_4 (Humans_Cars)/Car/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/ceiling_238_10 (Humans_Gym balls)/Ball/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/combined_5 (Humans_Dogs)/Dog/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/kh_3 (Humans_Gym balls)/Dog/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/prb_2 (Humans_Gym balls)/Dog/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/osu_farm_meadow_may24-28_2016_subset_113 (Cattle)/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Radar_site1_hilltop (Cattle)/',
    '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Bumblebee/Targets/Radar_site2_creamery_subset_113 (Cattle)/'
]

sampling_rate = 256

win_lens = [1, 1.5, 2, 2.5, 3]

######################### KEEP THE REST INTACT #########################

for winlen in [int(x * sampling_rate) for x in win_lens]:
    # Create windows
    print('----------------Human BumbleBee Targets (winlen={})----------------'.format(winlen))
    # Bumblebee human cuts
    extract_windows(indirs=human_dirs,
                    outdir=outdir,
                    class_label='Human',
                    stride=winlen,
                    winlen=winlen,
                    windows_all=True,
                    minlen_secs=winlen / sampling_rate)

    print('----------------Non-human BumbleBee Targets (winlen={})----------------'.format(winlen))

    # Bumblebee nonhuman cuts
    extract_windows(indirs=nonhuman_dirs,
                    outdir=outdir,
                    class_label='Nonhuman',
                    stride=winlen,
                    winlen=winlen,
                    windows_all=True,
                    minlen_secs=winlen / sampling_rate)

    # Create CV folds
    print('###################### Creating train-val-test splits (winlen={}) #####################'.format(winlen))
    sys.argv = init_argv + ['-type', prefix,
                            '-base', os.path.join(outdir, 'winlen_' + str(winlen) + '_winindex_all'),
                            '-outdir', foldoutdir,
                            '-cldirs', classdirs]

    if not(reload_flag):
        import Scripts.create_train_val_test_split
        reload_flag = True
    else:
        reload(Scripts.create_train_val_test_split)

    print('Extraction completed.\n')
