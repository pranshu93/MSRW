outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
