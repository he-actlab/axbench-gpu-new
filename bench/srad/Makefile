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
MODULE := srad_$(SIM_TYPE).out


# link objects(binaries) together
a.out:	DIR ./obj/main.o
	$(CC)	./obj/main.o \
				-I$(INCLUDE_BIN) \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o ./bin/$(MODULE)

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
./obj/main.o: 	./src/main.cu \
				./src/define.c \
				./src/graphics.c \
				./src/extract_kernel.cu \
				./src/prepare_kernel.cu \
				./src/reduce_kernel.cu \
				./src/srad2_kernel.cu \
				./src/compress_kernel.cu
	nvcc	$(CFLAGS) $(INCLUDE) -D$(SIM_TYPE) -I$(NVIDIA_HELPER) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -O3 -arch sm_20 -c ./src/main.cu -o ./obj/main.o


# delete all object files
clean:
	rm -rf ./bin/$(MODULE) 
	rm -rf ./obj/*.*

