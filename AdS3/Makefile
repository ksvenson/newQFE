#Dependency and compiler preamble
#============================================================
#Your path to ARPACK
#ARPACK=/usr/local/Cellar/arpack/3.6.2
ARPACK=/usr/local/Cellar/arpack/3.7.0_3
ARPACK_LIB= -L${ARPACK}/lib/ -larpack
ARPACK_FLAGS= -DUSE_ARPACK

#Your path to Eigen
#EIGEN=/share/pkg.7/eigen/3.3.5/src/eigen-eigen-b3f3d4950030
#EIGEN=$(CURDIR)/Eigen
EIGEN=/Users/hattrick/Desktop/Dean/adsSpectrum/AdS2-Lattice/AdS3/eigen-3.3.7/
#DIR = $(dir)
#EIGEN=DIR/eigen-3.37/

CXX = g++ -std=c++11
CXXFLAGS = -O2 -g -Wall -std=c++11 -I${EIGEN} -I. -I/usr/local/include -Wall -Wno-sign-compare 

LIBDIRS = -lgsl ${ARPACK_LIB}
SPECTRUM_INCLUDES = graph.h util.h cg.h cg_multishift.h eigen.h graph.h #arpack_interface_ads.h
#============================================================

#spectrum executable
#============================================================
all: spectrum

spectrum: ads_graph.o
	$(CXX) $(CXXFLAGS) -o spectrum ads_graph.o $(LIBDIRS) 

ads_graph.o: ads_graph.cpp Makefile ${SPECTRUM_INCLUDES}
	$(CXX) $(CXXFLAGS) -c ads_graph.cpp
#============================================================

#deltalin executable
#============================================================
#all: deltalin

#deltalin: delta_lin_fit.o
#	$(CXX) $(CXXFLAGS) -o deltalin delta_lin_fit.o  $(LIBDIRS) 

#delta_lin_fit.o: delta_lin_fit.cpp Makefile 
#	$(CXX) $(CXXFLAGS) -c delta_lin_fit.cpp
#============================================================

#clean
#============================================================
ALL_SOURCES = Makefile ads_graph.cpp $(SPECTRUM_INCLUDES)

clean:
	rm -f spectrum *.o core*
#============================================================
