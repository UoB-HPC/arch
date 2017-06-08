
all:
	nvcc test.cu -ccbin g++ -Xcompiler "-O3 -fopenmp" -g -arch=sm_35 -o stagin

clean:
	rm -rf stagin
