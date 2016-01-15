include ../config.mk

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
CUDA_LIB_DIR64 := $(CUDA_DIR)/lib64
CC := $(CUDA_DIR)/bin/nvcc
CFLAGS		:=
LFLAGS		:= 
HEADERS     := src
INCLUDE 	:= -I${HEADERS} -I${NVIDIA_HELPER}
LIB			:=

INCLUDE_BIN := $(CUDA_DIR)/include
SIM_TYPE := orig_code
MODULE := sobel_${SIM_TYPE}.out

# link objects(binaries) together
a.out:		DIR ./obj/sobel.o
			$(CC)	./obj/sobel.o \
					-I$(INCLUDE_BIN) \
					-L$(CUDA_LIB_DIR64) \
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
./obj/sobel.o: 	./src/CudaFilter.cu
	nvcc	./src/CudaFilter.cu \
				$(CFLAGS) $(INCLUDE) -O3 -I$(NVIDIA_HELPER) -D$(SIM_TYPE) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c -arch sm_20 -o ./obj/sobel.o

# delete all object files
clean:
	rm -rf ./bin/sobel_$(SIM_TYPE).out 
	rm -rf ./obj/sobel.o

