include ../config.mk

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
CUDA_LIB_DIR64 := $(CUDA_DIR)/lib64
CC := $(CUDA_DIR)/bin/nvcc
CFLAGS		:=
LFLAGS		:= 
HEADERS     := src
INCLUDE 	:= -I${HEADERS} -I/home/yazdan/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIB			:=

INCLUDE_BIN := $(CUDA_DIR)/include
SIM_TYPE := orig_code
MODULE := meanfilter_${SIM_TYPE}.out

# link objects(binaries) together
a.out:		DIR ./obj/meanFilter.o
			$(CC)	./obj/meanFilter.o \
					-I$(INCLUDE_BIN) \
					-L$(CUDA_LIB_DIR)  \
					-O3 -D$(SIM_TYPE) -lm -lcuda -lcudart -o ./bin/$(MODULE)

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi

# compile main function file into object (binary)
./obj/meanFilter.o: 	./src/CudaFilter.cu
	nvcc	./src/CudaFilter.cu \
				$(CFLAGS) $(INCLUDE) -O3 -I$(NVIDIA_HELPER) -D$(SIM_TYPE) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c -arch sm_20 -o ./obj/meanFilter.o

# delete all object files
clean:
	rm -rf ./bin/meanFilter_$(SIM_TYPE).out 
	rm -rf ./obj/meanFilter.o

