dir=$1
out_files=(`find $dir -name "*spl.out"`)

for file in ${out_files[@]}; do
    wc_op=(`wc -l $file`)
    left=$((12-${wc_op[0]}))
    fname=${wc_op[1]}
    if [ $left -gt 0 ]
    then
	sh_file=$(echo $(basename "$fname") | sed -e 's/spl.out/spl.sh/')
	restart_file=$(echo $(basename "$fname") | sed -e 's/spl.out/restart_spl.sh/') 
	echo ${restart_file}
	echo 'outname=`echo $0 | sed "s/.sh/.out/g"`' > ${restart_file}
	tail -n $left ${sh_file} >> ${restart_file}
    fi
done
