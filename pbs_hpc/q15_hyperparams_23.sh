outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 2 -q15 True -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 3 -q15 True -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 4 -q15 True -out $outname
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantTanh -ur 0.75 -wr 0.75 -w 32 -sp 0.5 -lr 0.005 -bs 128 -hs 16 -ot 0 -ml 768 -fn 5 -q15 True -out $outname
