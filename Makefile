CC = nvcc 
FLAGS =
OBJ = *.o
EXE = cuda_jacobi_exec
EXE1 = cuda_gs_rb_exec
EXE2 = cuda_jacobi_float_exec
EXE3 = cuda_gs_rb_float_exec

all: ${EXE} ${EXE1} ${EXE2} ${EXE3}

cuda_jacobi_float_exec: poisson_cuda_jacobi_float.cu
	$(CC) -o $@ $^ 
cuda_jacobi_exec: poisson_cuda_jacobi.cu
	$(CC) -o $@ $^ 
cuda_gs_rb_exec: poisson_cuda_gs_rb.cu
	$(CC) -o $@ $^ 
cuda_gs_rb_float_exec: poisson_cuda_gs_rb_float.cu
	$(CC) -o $@ $^ 
clean:
	rm -f $(OBJ) $(EXE) $(EXE1) $(EXE2) $(EXE3)