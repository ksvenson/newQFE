CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall

OBJ=lattice.o rng.o statistics.o phi4.o ising.o
BIN_DIR=../bin

all: $(addprefix $(BIN_DIR)/, lattice_test overrelax_test ising_test)

%.o: %.cc %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PRECIOUS: %.o

$(BIN_DIR)/%: %.cc $(OBJ)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OBJ) $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJ) *~
	$(RM) -r $(BIN_DIR)
