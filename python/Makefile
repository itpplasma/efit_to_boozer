# Name of the generated Python module
MODULE_NAME = efit2boozer

# Directory containing Fortran source files
SRC_DIR = ../SRC

# Specific Fortran source files

FORTRAN_FILES = efit_to_boozer_python.f90 field_divB0.f90 \
	efit_to_boozer_mod.f90 \
	spline5_RZ.f90 spl_three_to_five_mod.f90 \
	bdivfree_coul.f90 \
	field_line_integration_for_Boozer.f90 \
	plag_coeff.f90 binsrc.f90 rhs.f90 \
	spline_and_interpolate_magdata.f90
FORTRAN_SOURCES = $(addprefix $(SRC_DIR)/,$(FORTRAN_FILES))

EXTRA_OBJECTS = ../OBJS/bdivfree_coul.o \
	../OBJS/odeint_allroutines.o

# Target for building the f2py module
$(MODULE_NAME).so: $(FORTRAN_SOURCES)
	f2py -c -m $(MODULE_NAME) $(FORTRAN_SOURCES) $(EXTRA_OBJECTS) -I../OBJS -lblas -llapack

# Clean target
clean:
	rm -f *.mod *.o *.so
	rm -f ../OBJS/*.o ../OBJS/*.mod
