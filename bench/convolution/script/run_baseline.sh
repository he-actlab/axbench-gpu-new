#!/bin/bash

# Amir Yazdanbakhsh
# March 3, 2015

. ../include/bash_color.h
. ../include/gpgpu_sim.mk


# Global Configurations
MICROARCH=GTX480
BENCHMARK=convolutionSeparable
GPGPUSIM_CONFIG_DIR=gtx480_baseline
SIMULATION_TYPE=orig_code
BIN=${BENCHMARK}_${SIMULATION_TYPE}.out
EXTENSION=${GPGPUSIM_CONFIG_DIR}_${SIMULATION_TYPE}
LOG_DIR=baseline

Usage()
{
	echo -e "${Red}./run.sh${White}"
}

if [ "$#" -ge 1 ]; then
	if [ "$1" == "--help" ]; then
		Usage
	else
		echo -e "${Red}Use --help to learn how to use this bash script.${White}"
	fi
else
	echo -e "${Blue}# create the log directory ${White}"
		if [ ! -d log ]; then
			mkdir log
		fi
		if [ ! -d log/${LOG_DIR} ]; then
			mkdir log/${LOG_DIR}
		fi
		if [ ! -d log/${LOG_DIR}/${EXTENSION} ]; then
			mkdir log/${LOG_DIR}/${EXTENSION}
		fi

		if [ ! -d exec_dir ]; then
			mkdir exec_dir
		fi
		if [ ! -d exec_dir/${EXTENSION} ]; then
			mkdir exec_dir/${EXTENSION} 
		fi

	echo -e "${Blue}# copying the ${MICROARCH} config files...${White}"
		cp ${SIM_DIR}/configs/${MICROARCH}/${GPGPUSIM_CONFIG_DIR}/* ./exec_dir/${EXTENSION}


	echo -e "${Blue}# make the source file...${White}"
		make clean > /dev/null
		make SIM_TYPE=${SIMULATION_TYPE} > make_log 2>&1
		if [ "$?" -ne 0 ]; then
			echo -e"${Red} Build failed...${White}"
			cat make_log
			exit
		fi

	echo -e "${Blue}# copying the binary into the execution folder...${White}"
		cp ./bin/${BIN} ./exec_dir/${EXTENSION}
		cp ./ptx/${BENCHMARK}_${SIMULATION_TYPE}.ptx ./exec_dir/${EXTENSION}

	echo -e "${Red}# export PTX variable...${White}"
		export PTX_SIM_USE_PTX_FILE=1
		export PTX_SIM_KERNELFILE="${BENCHMARK}_${SIMULATION_TYPE}.ptx"

	echo -e "${Red}# GPGPU-Sim will use ${PTX_SIM_KERNELFILE} for simulation...${White}"

	echo -e "${Blue}# run the GPGPU-Sim...${White}"
		cd ./exec_dir/${EXTENSION}
		for f in ../../data/input/*.pgm
		do
			filename=$(basename "$f")
			extension="${filename##*.}"
			filename="${filename%.*}"

			rm -rf gpgpusim_power_report__*.log
			./${BIN} $f > ../../log/${LOG_DIR}/${EXTENSION}/${filename}_${EXTENSION}.log
			echo -e "${Blue}# Creating the power log file... ${White}"
			cp gpgpusim_power_report__*.log ../../log/${LOG_DIR}/${EXTENSION}/${filename}_${EXTENSION}.pwr
		done
fi