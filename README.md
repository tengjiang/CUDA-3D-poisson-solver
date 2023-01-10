# Cuda-based Acceleration for Solving Poisson Equation
Removing or changing some of the dependencies of the Gauss-Seidel method to facilitate the parallel use of CUDA to solve the Poisson system of equations on a 3D rectangular grid. 

Please see the report for more details. (In Chinese, English version to be added.)

## Highlights:
* Optimized binary tree parallel reduction in CUDA up to a warp level. Utilized techniques such as sequential addressing, first add during load, and warp unrolling.
* Numerically computed Poisson equation with Jacobi method, Red & Black Gauss-Seidel method, and Block-wise Gauss-Seidel method on GPU. Tuned hyper-parameters with grid search and optimized memory access with 3-D texture memory. Enhanced the time/bandwidth efficiency by 26 times.

![CUDA poster](https://user-images.githubusercontent.com/45305359/211681534-d7c2b59c-a6fb-4bba-9a40-530a9e777f6f.jpg)

