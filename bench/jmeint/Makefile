include ../config.mk
INCLUDES  := -I../../common/inc


CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
CUDA_LIB_DIR64 := $(CUDA_DIR)/lib64

HEADERS     := src
INCLUDE 	:= -I${HEADERS} -I/usr/local/cuda/include
LIB			:= -L/usr/local/cuda/lib64
SIM_TYPE 	:= orig_code
MODULE		:= jmeint_$(SIM_TYPE).out 
LIB_CUDA	:= -lcuda -lcudart
#DEBUG      := -g -G -O0
################################################################################

# Target rules
all: DIR build

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi


build: $(MODULE)

./obj/jmeint.o:./src/main.cu
	nvcc $(CFLAGS) $(INCLUDE) -O3 -I$(NVIDIA_HELPER) $(INCLUDES) -D$(SIM_TYPE) $(ALL_CCFLAGS) -gencode arch=compute_20,code=compute_20 $(LIB_CUDA) -c $< -o $@

$(MODULE): ./obj/jmeint.o
	nvcc -O3 $(INCLUDE) -D$(SIM_TYPE) $(LIB) -o ./bin/$@ $+

clean:
	rm -f ./bin/$(MODULE)
	rm -f ./obj/jmeint.o

clobber: clean
