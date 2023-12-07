FC = gfortran -fbounds-check
target = esegui
sources = $(wildcard *.f90)
objs = $(sources:.f90=.o)
libs = -llapack -lblas 


all:
	$(FC) -c $(sources)
	$(FC) -o $(target) $(objs) $(libs)




