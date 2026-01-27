#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
int g_block_size = 32;

int getBlockSize() {
  const char* env = std::getenv("BLOCK_SIZE");
  if (env && *env) {
    int val = std::atoi(env);
    if (val > 0)
      return val;
  }
  return g_block_size;
}

/*void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); ++j)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); ++k)
      for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
        C(i, j) += A(i, k) * B(k, j);

}*/

// Orden optimizado: i, k, j
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i) {
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); ++k) {
      for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

}  // namespace

void setBlockSize(int size) {
  if (size > 0)
    g_block_size = size;
}

void setNbThreads(int n) {
#if defined(_OPENMP)
  if (n > 0)
    omp_set_num_threads(n);
#else
  (void)n;
#endif
}

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  const int block = getBlockSize();
#if defined(_OPENMP)
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int i = 0; i < A.nbRows; i += block)
    for (int j = 0; j < B.nbCols; j += block)
      for (int k = 0; k < A.nbCols; k += block)
        prodSubBlocks(i, j, k, block, A, B, C);
  return C;
}
