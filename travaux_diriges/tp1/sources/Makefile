include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

MPICC ?= mpicc -fopenmp

CFLAGS = -std=c11
CXXFLAGS = -std=c++14
ifdef DEBUG
CFLAGS += -g -O0 -Wall -pedantic
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = ${CXXFLAGS}
else
CFLAGS += -O2 -march=native -Wall
CXXFLAGS += -O2 -march=native -Wall
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
endif

ALL= calcul_pi.exe TestProductMatrix.exe test_product_matrice_blas.exe jeton.exe pi_mpi.exe hypercube.exe hypercube_seq.exe

default:    help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ Output*.txt

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $^ -o $@  

calcul_pi.exe: calcul_pi.cpp
	$(MPICXX) $(CXXFLAGS) $^ -o $@

pi_mpi.exe: pi_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $^ -o $@

TestProductMatrix.exe : TestProductMatrix.o Matrix.hpp Matrix.o ProdMatMat.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB)  

test_product_matrice_blas.exe : test_product_matrice_blas.o Matrix.hpp Matrix.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB)  $(BLAS)

jeton.exe: jeton.c
	$(MPICC) $(CFLAGS) $^ -o $@

hypercube.exe: hypercube.c
	$(MPICC) $(CFLAGS) $^ -o $@

hypercube_seq.exe: hypercube_seq.c
	$(MPICC) $(CFLAGS) $^ -o $@

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
