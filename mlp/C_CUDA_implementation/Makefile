CC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_PATH ?= /opt/apps/rhel9/cuda-11.8
CFLAGS = -std=c++11 -I$(CUDA_PATH)/include
NVCCFLAGS = -arch=sm_80

all: mlp

mlp: main.o mlp.o kernels.o
	$(NVCC) $(NVCCFLAGS) -o mlp main.o mlp.o kernels.o

main.o: main.cpp mlp.h
	$(CC) $(CFLAGS) -c main.cpp

mlp.o: mlp.cpp mlp.h kernels.h
	$(CC) $(CFLAGS) -c mlp.cpp

kernels.o: kernels.cu kernels.h
	$(NVCC) $(NVCCFLAGS) -c kernels.cu

clean:
	rm -f *.o mlp
