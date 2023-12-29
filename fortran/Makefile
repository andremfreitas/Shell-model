FC = gfortran -fbounds-check
target = exe
sources = precision.f90 Integration.f90  main.f90     
objs = $(sources:.f90=.o)
mods = $(sources:.f90=.mod)
libs = -llapack -lblas 


all:
	$(FC) -c precision.f90
	$(FC) -c Integration.f90
	$(FC) -c main.f90

	$(FC) -o $(target) $(objs) $(libs)

clean:
	rm -f *.mod *.csv $(target) $(objs)