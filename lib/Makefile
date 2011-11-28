CC = g++ 
LOPT = -lm
AR = ar rcv
RANLIB = ranlib

HEADERS = vectors.h
FILES   = vectors.o

lib: $(HEADERS) $(FILES)
	$(AR) lib.a $(FILES)
	$(RANLIB) lib.a

clean:
	rm -f *~ *.o *.a

.cc.o:
	$(CC) $(CFLAG) -c $*.cc
