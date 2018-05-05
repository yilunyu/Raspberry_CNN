all: main.cpp operation.cpp operation.h tensor.cpp tensor.h controller.cpp controller.h
	g++ -Wall -std=c++11 -O2 -mfpu=neon -fno-tree-vectorize main.cpp controller.cpp operation.cpp tensor.cpp -I/home/pi/Ne10/inc libNE10.a #-lm
