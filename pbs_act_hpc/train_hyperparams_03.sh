outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantTanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 1 -type act -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantTanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 2 -type act -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantTanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 3 -type act -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantTanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 4 -type act -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantSigm -gunl quantTanh -ur 0.25 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 5 -type act -out $outname
