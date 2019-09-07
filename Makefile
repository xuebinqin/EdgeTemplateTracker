# definitions
objRRC = STPF_main.o CommonFunctions.o
srcRRC = STPF_main.cpp CommonFunctions.cpp

#linker to use
lnk = g++
#compiler to use
cc = gcc
#uncomment for debugging
dbg = -g -Wall

# MAKE it happen

all: STPF_main

STPF_main: $(objRRC)
	$(lnk) $(dbg) -o STPF_main $(objRRC) EDLib.a EDLinesLib.a `pkg-config --libs opencv`

$(objRRC): $(srcRRC)
	$(cc) $(dbg) `pkg-config --cflags opencv` -c $(srcRRC)

clean:
	@rm -f $(objRRC) STPF_main


#all:
#	gcc `pkg-config --cflags opencv` -o VideoEDLines $(srcRRC) EDLinesLib.a libopencv_imgproc.so.2.4.5 libopencv_core.so.2.4.5 `pkg-config --libs opencv`

