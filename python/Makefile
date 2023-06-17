# Name of the generated Python module
MODULE_NAME = efit2boozer

# Directory containing Fortran source files
SRC_DIR = ../SRC

# Specific Fortran source files

FORTRAN_FILES = field_divB0.f90 efit_to_boozer_mod.f90 \
	spline5_RZ.f90 spl_three_to_five_mod.f90 \
	field_line_integration_for_Boozer.f90 \
	plag_coeff.f90 binsrc.f90 rhs.f90 \
	spline_and_interpolate_magdata.f90
FORTRAN_SOURCES = $(addprefix $(SRC_DIR)/,$(FORTRAN_FILES))

EXTRA_FILES = bdivfree_coul.f90 \
	odeint_allroutines.f

EXTRA_SOURCES = $(addprefix $(SRC_DIR)/,$(EXTRA_FILES))
EXTRA_OBJECTS := $(EXTRA_FILES:.f90=.o)
EXTRA_OBJECTS := $(EXTRA_OBJECTS:.f=.o)

# Target for building the f2py module
$(MODULE_NAME).so: $(FORTRAN_SOURCES) $(EXTRA_SOURCES)
	gfortran -c $(EXTRA_SOURCES)
	f2py -c -m $(MODULE_NAME) $(FORTRAN_SOURCES) $(EXTRA_OBJECTS)

# Clean target
clean:
	rm -f *.mod *.o *.so
