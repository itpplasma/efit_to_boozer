add_library(efit_to_boozer SHARED
    SRC/odeint_allroutines.f
    SRC/efit_to_boozer.f90
    SRC/field_divB0.f90
    SRC/efit_to_boozer_mod.f90
    SRC/spline5_RZ.f90
    SRC/spl_three_to_five_mod.f90
    SRC/bdivfree_coul.f90
    SRC/field_line_integration_for_Boozer.f90
    SRC/plag_coeff.f90
    SRC/binsrc.f90
    SRC/rhs.f90
    SRC/spline_and_interpolate_magdata.f90
)
