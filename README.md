This GitHub repository is meant to act as a portfolio for code written by Rachel Clune.

Breif summary of the code included: 

1. hand_coded_abb_to_abb_beta_fock_scheme6.cpp: Example of thread parallelized linear algebra code that is part of a larger project for ab initio calculations of excited state properties. For more details on the project see https://doi.org/10.1021/acs.jctc.0c00308 This particular piece of the project performs a piece of the linear transformation needed to solve for the first-order wave function parameters for the perturbative method. Originally each piece of the linear transformation was written as a series of for loops and conditionals, but due to the massive size of the tensors involved (for this piece of code the largest tensors are rank-6) converting the code into matrix math would reduce the computational cost of the code significantly. To do this we broke down the math involved into pieces based on the conditional statements and used Intel's batched GEMM tool include in MKL to significantly speed up the calculation. 

2. Code contained in the internally_contracted_esmp2 folder: The pieces of code here are portions of a previous attempt at the project detailed in (1). The use of an internally contracted wave function led to linear dependencies that made the method unsuable. These pieces of code are written fully in Python, with some autogeneration tools used to convert some of the mathematical operations in to Fortran code to increase the computation speed. 

3. Code contained in the TDHF/RPA folder: Code I wrote as part of an assignment for a graduate course I was taking in my first year at UC Berkeley that calculates the excited state energies of small molecules using time-dependent Hartree Fock.
