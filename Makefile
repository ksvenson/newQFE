CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall

OBJ=src/lattice.o src/wolff.o src/rng.o src/overrelax.o src/metropolis.o \
		src/statistics.o
BIN=bin/lattice_test bin/overrelax_test

all: ${BIN}

%.o: %.cc %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/lattice_test: src/lattice_test.cc $(OBJ)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJ) $< -o $@

bin/overrelax_test: src/overrelax_test.cc $(OBJ)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJ) $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJ) *~
	$(RM) -r bin
