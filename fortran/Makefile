FC = gfortran -fbounds-check
target = exe
sources = $(wildcard *.f90)
objs = $(sources:.f90=.o)
mods = $(sources:.f90=.mod)
libs = -llapack -lblas 


all:
	$(FC) -c $(sources)
	$(FC) -o $(target) $(objs) $(libs)

clean:
	rm -f *.mod *.csv $(target) $(objs)