outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
python3 ../Architecture/dynamic_rnn_tvt.py -ggnl tanh -gunl tanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -type embedding -q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_640_winindex_all/HumanVsNonhuman_48_16/embedding_H=32_k=38_ep=10_it=10_rnd=10 -out $outname
