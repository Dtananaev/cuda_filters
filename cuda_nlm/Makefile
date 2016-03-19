project: nlm.cc Makefile
	nvcc -O3 -Xcompiler="-fopenmp" -arch=compute_20 -code=compute_20 -I. nlm.cc cuda_nlm.cu cuda_nlm_smem.cu timer.cc -o cunlm
clean:
	rm cunlm
