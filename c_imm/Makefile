# Two-target Makefile: identical filter/main sources, swapped linalg backend.
#
#   make pure   -> imm_pure.exe (portable C, no external math lib)
#   make blas   -> imm_blas.exe (links OpenBLAS via -lopenblas)
#   make all    -> both
#
# Tune optimization with `make CFLAGS_EXTRA=-fno-tree-vectorize` etc.

CC        ?= gcc
CFLAGS     = -O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-parameter \
             -ffast-math -funroll-loops $(CFLAGS_EXTRA)
LDLIBS     = -lm

COMMON_SRC = filters.c main.c
COMMON_HDR = filters.h linalg.h

# Backend variants
PURE_SRC   = $(COMMON_SRC) linalg_pure.c
BLAS_SRC   = $(COMMON_SRC) linalg_blas.c

BLAS_LDLIBS = -lopenblas $(LDLIBS)

EXE_PURE = imm_pure.exe
EXE_BLAS = imm_blas.exe

.PHONY: all pure blas clean

all: pure blas

pure: $(EXE_PURE)

blas: $(EXE_BLAS)

$(EXE_PURE): $(PURE_SRC) $(COMMON_HDR)
	$(CC) $(CFLAGS) -DBACKEND_LABEL=\"pure-c\" -o $@ $(PURE_SRC) $(LDLIBS)

$(EXE_BLAS): $(BLAS_SRC) $(COMMON_HDR)
	$(CC) $(CFLAGS) -DBACKEND_LABEL=\"openblas\" -o $@ $(BLAS_SRC) $(BLAS_LDLIBS)

clean:
	rm -f $(EXE_PURE) $(EXE_BLAS) *.o
