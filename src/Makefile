CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall

OBJ=lattice.o wolff.o rng.o overrelax.o metropolis.o statistics.o
BIN_DIR=../bin

all: $(addprefix $(BIN_DIR)/, lattice_test overrelax_test)

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
