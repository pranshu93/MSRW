#!/usr/bin/env bash
###############################################################################################################################################################
# Hyperparameters: ref: https://github.com/Microsoft/EdgeML/blob/master/docs/README_BONSAI_OSS.md
###############################################################################################################################################################
# Bonsai [Options] DataFolder
# Options:
#
# -F    : [Required] Number of features in the data.
# -C    : [Required] Number of Classification Classes/Labels.
# -nT   : [Required] Number of training examples.
# -nE   : [Required] Number of examples in test file.
# -f    : [Optional] Input format. Takes two values [0 and 1]. 0 is for libsvm_format(default), 1 is for tab/space separated input.
#
# -P   : [Optional] Projection Dimension. (Default: 10 Try: [5, 20, 30, 50])
# -D   : [Optional] Depth of the Bonsai tree. (Default: 3 Try: [2, 4, 5])
# -S   : [Optional] sigma = parameter for sigmoid sharpness  (Default: 1.0 Try: [3.0, 0.05, 0.005] ).
#
# -lW  : [Optional] lW = regularizer for predictor parameter W  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
# -lT  : [Optional] lTheta = regularizer for branching parameter Theta  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
# -lV  : [Optional] lV = regularizer for predictor parameter V  (Default: 0.0001 Try: [0.01, 0.001, 0.00001]).
# -lZ  : [Optional] lZ = regularizer for projection parameter Z  (Default: 0.00001 Try: [0.001, 0.0001, 0.000001]).
#
# Use Sparsity Params to vary your model size for a given tree depth and projection dimension
# -sW  : [Optional] lambdaW = sparsity for predictor parameter W  (Default: For Binaay 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
# -sT  : [Optional] lambdaTheta = sparsity for branching parameter Theta  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
# -sV  : [Optional] lambdaV = sparsity for predictor parameters V  (Default: For Binary 1.0 else 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
# -sZ  : [Optional] lambdaZ = sparsity for projection parameters Z  (Default: 0.2 Try: [0.1, 0.3, 0.4, 0.5]).
#
# -I   : [Optional] [Default: 42 Try: [100, 30, 60]] Number of passes through the dataset.
# -B   : [Optional] Batch Factor [Default: 1 Try: [2.5, 10, 100]] Float Factor to multiply with sqrt(ntrain) to make the batch_size = min(max(100, B*sqrt(nT)), nT).
# DataFolder : [Required] Path to folder containing data with filenames being 'train.txt' and 'test.txt' in the folder."
#
# Note - Both libsvm_format and Space/Tab separated format can be either Zero or One Indexed in labels. To use Zero Index enable ZERO_BASED_IO flag in config.mk
# and recompile Bonsai
###############################################################################################################################################################

########################################################
# Check number of args
########################################################
if [ $# -ne 1 ]
then
	echo "Usage: $0 <pathname>"
	exit 1
fi

# Overall parameter space #
P=(5)
D=(2 3)
S=(1.0)
lW=(0.00001 0.0001 0.001 0.01)
lV=(0.00001 0.0001 0.001 0.01)
lT=(0.00001 0.0001 0.001 0.01)
lZ=(0.000001 0.00001 0.0001 0.001)
sZ=(0.1 0.2 0.3 0.4 0.5)

# Output filename
echo 'outname=`echo $0 | sed "s/.sh/.out/g"`'

for i in ${P[@]};
do
	for j in ${D[@]};
	do
        for k in ${S[@]};
        do
            for l in ${lW[@]};
            do
                for m in ${lV[@]};
                do
                    for n in ${lT[@]};
                    do
                        for o in ${lZ[@]};
                        do
                            for p in ${sZ[@]};
                            do
                                for r in `seq 1 5`;
                                do
                                    echo python3 ../Architecture/dynamic_rnn_bonsai.py -base $1 -ct 1 -w 32 -sp 0.5 -lr 0.005 - bs 128 -hs 16 -ot 1 -ml 768 -fn $r -P $i -dep $j -sig $k -rW $l -rV $m -rT $n -rZ $o -sZ $p -out '$outname'
                                done
                            done
                        done
                    done
                done
            done
        done
	done
done