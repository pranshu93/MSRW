from sklearn.model_selection import KFold
import numpy as np
import argparse
import os

# Set random seed
np.random.seed(42)

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('-type', type=str, default='bb_tar', help='Classification type: \'tar\' for target,' \
                                                                   ' \'act\' for activity)')
parser.add_argument('-base', type=str, default='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/'
                                               'Data/Bumblebee/Windowed/winlen_384_winindex_all',
                    help='Base location of data')
parser.add_argument('-outdir', type=str, default='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/'
                                                 'FastGRNN/Data/Bumblebee', help='Output folder')
parser.add_argument('-cldirs', type=list, default=['Human', 'Nonhuman'], help='Class folder paths relative to base')
#parser.add_argument('-hum', type=str, default='Human', help='Human cuts folder relative to base')
#parser.add_argument('-nhum', type=str, default='Nonhuman', help='Nonhuman cuts folder relative to base')
parser.add_argument('-nfolds', type=int, default=5, help='Number of cross-validation folds')

args=parser.parse_args()

if args.outdir == None:
    outdir = args.base
else:
    outdir = args.outdir
# Silently create output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

fileloc = args.base
#filestrs = [args.hum, args.nhum]
filestrs = args.cldirs

data = [];
labels = [];
[[data.append(np.fromfile(open(os.path.join(os.path.join(fileloc, filestr), filename), "r"), dtype=np.uint16).tolist())
  for filename in os.listdir(os.path.join(fileloc, filestr))] for filestr in filestrs];
data = np.array(data, dtype=object);
[labels.extend([i] * [None for filename in os.listdir(os.path.join(fileloc, filestrs[i]))].__len__()) for i in
 range(filestrs.__len__())]

data = np.array(data)
print('Total number of data points: {}'.format(data.__len__()))
labels = np.array(labels)

# Get splits
kf = KFold(n_splits=args.nfolds, random_state=42, shuffle=True)

# Prefix
prefix = args.type + '_' + os.path.basename(args.base)

# Create output folder
if not(os.path.exists(os.path.join(outdir, prefix))):
    os.mkdir(os.path.join(outdir, prefix))

i=0
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Save train data
    np.save(os.path.join(outdir, prefix, prefix + str(i) + "_train.npy"), X_train)
    np.save(os.path.join(outdir, prefix, prefix + str(i) + "_train_lbls.npy"), y_train)

    # Save test data
    np.save(os.path.join(outdir, prefix, prefix + str(i) + "_test.npy"), X_test)
    np.save(os.path.join(outdir, prefix, prefix + str(i) + "_test_lbls.npy"), y_test)

    i += 1

