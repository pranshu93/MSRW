outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 32 -sp 1 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 32 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 32 -hs 48 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 32 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 64 -hs 48 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.5 -lr 0.005 -bs 128 -hs 48 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 3 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 4 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 16 -ot 1 -ml 768 -fn 5 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 1 -q15 True -bat slurm -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantSigm -ur 0.25 -wr 0.25 -w 64 -sp 0.75 -lr 0.005 -bs 32 -hs 32 -ot 0 -ml 768 -fn 2 -q15 True -bat slurm -out $outname
