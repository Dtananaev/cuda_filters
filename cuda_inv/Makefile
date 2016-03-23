cuinv: cuinv.cu Makefile
	nvcc -gencode=arch=compute_20,code=compute_20 -gencode=arch=compute_20,code=sm_20 -I. cuinv.cu timer.cc -o cuinv
clean:
	rm cuinv
