outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 1 -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 2 -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 3 -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 4 -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.25 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 32 -ot 0 -ml 768 -fn 5 -out $outname
