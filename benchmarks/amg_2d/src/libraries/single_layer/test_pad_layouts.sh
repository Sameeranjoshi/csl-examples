#!/bin/bash

echo "
Testing with square matrices and matching kernel sizes:(divisible)"
python jacobi_only.py 4 4 2 2   # larger
python jacobi_only.py 2 2 2 2   # equal
python jacobi_only.py 2 2 4 4 # smaller than kernel


echo "
Testing with rectangular matrices:(divisible but not square)"
python jacobi_only.py 6 4 2 2
python jacobi_only.py 4 6 2 2

echo "
Testing with different kernel dimensions:(divisible but not square)"
python jacobi_only.py 6 6 2 3
python jacobi_only.py 6 6 3 2

echo "
Testing with matrix size greater than kernel size:"
echo "
Testing with non-divisible sizes:(non-divisible and not square)"
python jacobi_only.py 7 7 3 3
python jacobi_only.py 10 9 4 3
python jacobi_only.py 11 13 3 4


echo "
Testing with matrix size smaller than kernel size:"
python jacobi_only.py 2 2 3 3
python jacobi_only.py 3 3 4 4
python jacobi_only.py 4 4 6 6
python jacobi_only.py 5 4 2 2
python jacobi_only.py 5 3 2 2