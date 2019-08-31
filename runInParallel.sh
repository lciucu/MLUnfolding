#!/bin/bash
#if there is no parameter, it stops and it gives the instructions
if [ $# -ne 0 ]; then
cat <<EOF
Usage: #0
EOF
exit 1
fi

DOTEST="0"

if [[ ${DOTEST} == "1" ]]; then
    ls="B3,B4,B5"
    ks="8"
    es="3"
    bs="1000"
else
    if false; then
	ls="B3"
	ks="4,8,16"
	es="3000"
	bs="200,1000"
    fi
    if false; then
	ls="A1,B2,B3,B4,B5,B10"
	ks="8"
	es="3000"
	bs="1000"
    fi
    if false; then
	ls="A1,B2,B3,B4,B5,B10"
	ks="4"
	es="3000"
	bs="1000"
    fi
    if false; then
	ls="A1,B2,B3,B4,B5,B10"
	ks="2"
	es="3000"
	bs="1000"
    fi
    if false; then
	ls="A1,B2,B3,B4,B5,B10"
	ks="1"
	es="3000"
	bs="1000"
    fi
    if false; then
	ls="A1,B2,B3,B4,B5,B10"
	ks="16"
	es="3000"
	bs="1000"
    fi
    if false; then
	ls="A1,B3"
	ks="8"
	es="3000"
	bs="60,200,1000,5000"
    fi
    
    # still running 05_NN_A1_8_3000_60 (1500/3000), (2700/3000) 05_NN_B3_8_3000_60, 04_NN_B10_16_3000_1000 (2500/3000), maybe others too

fi

STAGE="1110"
v="12"
folderVersion="05"

for l in `echo "${ls}" | awk -v RS=, '{print}'`
do
    for k in `echo "${ks}" | awk -v RS=, '{print}'`
    do
	for e in `echo "${es}" | awk -v RS=, '{print}'`
	do
	    for b in `echo "${bs}" | awk -v RS=, '{print}'`
	    do
		NN="${l}_${k}_${e}_${b}"
		echo "NN=${NN}"
		suffix="${folderVersion}_NN_${NN}"
		COMMAND="./MLUnfolding${v}.py ${NN} ${STAGE} output${v}_${suffix} >& run_${suffix}.log &"
		echo "COMMAND=${COMMAND}"
		eval ${COMMAND}
	    done
	done
    done
done

# e.g. ./MLUnfolding12.py  B3_8_3_1000,B4_8_3_200 1110 output11_7
