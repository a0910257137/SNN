#! /bin/sh
rm -rf ../backend/opencl/execution/bin/*.bin; 
make all; ./converter ../backend/opencl/execution/cl/;
mkdir ../backend/opencl/execution/bin; mv *.bin ../backend/opencl/execution/bin;