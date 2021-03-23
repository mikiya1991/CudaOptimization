CPPFLAGS:=-arch=sm_70 -I . -std=c++11 
TARGETS:=$(patsubst %.cu, %, $(wildcard *.cu))

all: $(TARGETS)

clean:
	-rm -rf *o 
	-rm $(TARGETS) 

$(TARGETS): % : %.o
	nvcc $^ -o $@

%.o: %.cu
	nvcc $(CPPFLAGS) -c $^ -o $@
