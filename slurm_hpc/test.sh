outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../Architecture/dynamic_rnn.py -ggnl quantTanh -gunl quantSigm -ur 0.75 -wr 0.75 -w 128 -sp 1 -lr 0.005 -bs 64 -hs 16 -ot 0 -ml 768 -fn 1 -q15 True -out $outname
