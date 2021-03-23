CPPFLAGS:= -I . -std=c++11 
TARGETS:=$(patsubst %.cpp, %, $(wildcard *.cpp))

all: $(TARGETS)

clean:
	-rm -rf *o 
	-rm $(TARGETS) 

$(TARGETS): % : %.o
	hipcc $^ -o $@

%.o: %.cpp
	hipcc $(CPPFLAGS) -c $^ -o $@
