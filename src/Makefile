CC = g++ 
#CC = clang++
FASTOPT = -O3 -march=native -Wall
DEBUGOPT = -O3 -pg -Wall

#CFLAG=$(FASTOPT)
CFLAG=$(DEBUGOPT)

all:eval-pct eval-pa eval-svmsgd eval-svmdcd

eval-pct:eval_pct.cc perceptron.o vectors.o
	$(CC) -o $@ $^ $(CFLAG) -I../lib -lm
eval-pa:eval_pa.cc passive_aggressive.o vectors.o
	$(CC) -o $@ $^ $(CFLAG) -I../lib -lm
eval-svmsgd:eval_svmsgd.cc svmsgd.o vectors.o
	$(CC) -o $@ $^ $(CFLAG) -I../lib -lm
eval-svmdcd:eval_svmdcd.cc svmdcd.o vectors.o
	$(CC) -o $@ $^ $(CFLAG) -I../lib -lm

perceptron.o:perceptron.cc perceptron.h
	$(CC) -o $@ -c perceptron.cc $(CFLAG) -I../lib
passive_aggressive.o:passive_aggressive.cc passive_aggressive.h
	$(CC) -o $@ -c passive_aggressive.cc $(CFLAG) -I../lib
svmsgd.o:svmsgd.cc svmsgd.h
	$(CC) -o $@ -c svmsgd.cc $(CFLAG) -I../lib
svmdcd.o:svmdcd.cc svmdcd.h
	$(CC) -o $@ -c svmdcd.cc $(CFLAG) -I../lib
vectors.o:../lib/vectors.cc ../lib/vectors.h
	$(CC) -o $@ -c ../lib/vectors.cc $(CFLAG) -I../lib

clean:
	rm -f *~ *.o
	rm -f eval-pct eval-pa eval-svmsgd eval-svmdcd

