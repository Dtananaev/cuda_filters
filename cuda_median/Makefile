cumed: cuda_median.cu Makefile
	nvcc -gencode=arch=compute_20,code=compute_20 -gencode=arch=compute_20,code=sm_20 -I. cuda_median.cu timer.cc -o cumed
clean:
	rm cumed
