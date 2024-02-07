CXX = gcc 
LDLIBS := -L/home2/anders/proj_c/SNN/third_party/OpenCL/ -lOpenCL
LDINCS := -I/home2/anders/proj_c/SNN/third_party/OpenCL/CL
CFLAGS  = -Wall -Wextra -pedantic -O3
FILE = clCompile.c
EXE := converter 
all: 
	$(CXX)  $(FILE)  -o  $(EXE)  $(LDLIBS) ${LDINCS}

clean:
	rm -rf $(OUT)