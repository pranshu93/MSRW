#!/usr/bin/env bash

# Assuming Scripts/get_specific_window_from_cut.py has run

cd ../Scripts
source activate tfgpu
python create_train_val_test_split.py -type pedbike_class -base /scratch/dr2915/austere/classification_data_windowed/winlen_256_winindex_all
python create_train_val_test_split.py -type pedbike_class -base /scratch/dr2915/austere/classification_data_windowed/winlen_384_winindex_all
python create_train_val_test_split.py -type pedbike_class -base /scratch/dr2915/austere/classification_data_windowed/winlen_512_winindex_all