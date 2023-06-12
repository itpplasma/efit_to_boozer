  program efit_to_boozer
!
  use efit_to_boozer_mod
  use input_files, only : gfile
!
  implicit none
!
  integer :: nstep,nsurfmax,i,is,it,nsurf,nt,mpol,inp_label,m,iunit
  double precision :: s,theta,hs,htheta,aiota,aJb,B_theta,B_phi,oneovernt,twoovernt,sqrtg00
  double precision :: psi,q,dq_ds,C_norm,dC_norm_ds,sqrtg,bmod,dbmod_dtheta,sigma,     &
                      R,dR_ds,dR_dtheta,Z,dZ_ds,dZ_dtheta,G,dG_ds,dG_dtheta
  double precision :: phi,Br,Bp,Bz,dBrdR,dBrdp,dBrdZ,dBpdR,dBpdp,dBpdZ,dBzdR,dBzdp,dBzdZ
  double precision :: dB_phi_ds,dB_theta_ds,pprime
  double complex   :: four_ampl
!
  double precision, dimension(:), allocatable :: R_oft,Z_oft,al_oft,B_oft
  double precision, dimension(:), allocatable :: Rmn_c,Rmn_s,Zmn_c,Zmn_s,almn_c,almn_s,Bmn_c,Bmn_s
  double complex,   dimension(:), allocatable :: calE,calEm_times_Jb
!
  open(1,file='efit_to_boozer.inp')
  read (1,*) nstep    !number of integration steps
  read (1,*) nlabel   !grid size over radial variable
  read (1,*) ntheta   !grid size over poloidal angle
  read (1,*) nsurfmax !number of starting points between the
                      !magnetic axis and right box boundary
                      !when searching for the separatrix
  read (1,*) nsurf    !number of flux surfaces in Boozer file
  read (1,*) mpol     !number of poloidal modes in Boozer file
  close(1)
!
  allocate(rbeg(nlabel),rsmall(nlabel),qsaf(nlabel),psi_pol(0:nlabel))
  allocate(psi_tor_vac(nlabel),psi_tor(0:nlabel),C_const(nlabel))
!
  allocate(R_spl(0:nspl,0:ntheta,nlabel),Z_spl(0:nspl,0:ntheta,nlabel),bmod_spl(0:nspl,0:ntheta,nlabel))
  allocate(sqgnorm_spl(0:nspl,0:ntheta,nlabel),Gfunc_spl(0:nspl,0:ntheta,nlabel))
!
  call flint_for_Boozer(nstep,nsurfmax,nlabel,ntheta,      &
                        rmn,rmx,zmn,zmx,raxis,zaxis,sigma, &
                        rbeg,rsmall,qsaf,                  &
                        psi_pol(1:nlabel),psi_tor_vac,     &
                        psi_tor(1:nlabel),C_const,         &
                        R_spl(0,1:ntheta,:),               &
                        Z_spl(0,1:ntheta,:),               &
                        bmod_spl(0,1:ntheta,:),            &
                        sqgnorm_spl(0,1:ntheta,:),         &
                        Gfunc_spl(0,1:ntheta,:))
!
  call spline_magdata_in_symfluxcoord
!
  print *,'Splining done'
!
  nt=ntheta
!
  inp_label=1
  hs=1.d0/dfloat(nsurf)
  oneovernt=1.d0/dfloat(nt)
  twoovernt=2.d0*oneovernt
  htheta=twopi*oneovernt
!
  allocate(calE(nt),calEm_times_Jb(nt))
  allocate(R_oft(nt),Z_oft(nt),al_oft(nt),B_oft(nt))
  allocate(Rmn_c(0:mpol),Rmn_s(0:mpol),Zmn_c(0:mpol),Zmn_s(0:mpol))
  allocate(almn_c(0:mpol),almn_s(0:mpol),Bmn_c(0:mpol),Bmn_s(0:mpol))
!
  iunit=71
  open(iunit,file='fromefit.bc')
  write(iunit,*) 'CC Boozer-coordinate data file generated from EFIT equilibrium file'
  write(iunit,*) 'CC Boozer file format: E. Strumberger'
  write(iunit,*) 'CC Authors: S.V. Kasilov, C.G. Albert'
  write(iunit,*) 'CC Original EFIT equilibrium file: ',trim(gfile)
  write(iunit,*) 'm0b   n0b  nsurf  nper    flux [Tm^2]        a [m]          R [m]'
  write(iunit,'(4i6,e15.6,2f10.5)') mpol, 0, nsurf, 1, sigma*psitor_max*1d-8*twopi, rsmall(nlabel)*1d-2, raxis*1d-2
!
  phi=0.d0
!
  do is=1,nsurf
    s=hs*(dfloat(is)-0.5d0)
    do it=1,nt
      theta=htheta*dfloat(it)
!
      call magdata_in_symfluxcoord_ext(inp_label,s,psi,theta,q,dq_ds,C_norm,dC_norm_ds,sqrtg,bmod,dbmod_dtheta, &
                                       R,dR_ds,dR_dtheta,Z,dZ_ds,dZ_dtheta,G,dG_ds,dG_dtheta)
!
      call field_eq(R,phi,Z,Br,Bp,Bz,dBrdR,dBrdp,dBrdZ,dBpdR,dBpdp,dBpdZ,dBzdR,dBzdp,dBzdZ)
!
      calE(it)=exp(cmplx(0.d0,theta+G/q))
      aJb=R*(Br**2+Bp**2+Bz**2)/(C_norm*Bp)
      calEm_times_Jb(it)=cmplx(aJb,0.d0)
!
      R_oft(it)=R
      Z_oft(it)=Z
      al_oft(it)=-G/twopi
      B_oft(it)=bmod
    enddo
!
! iota, poloidal and toroidal covariant components:
    aiota=1.d0/q
    B_phi=Bp*R
    B_theta=q*(C_norm-B_phi)
    sqrtg00=(B_phi+B_theta/q)*real(sum(calEm_times_Jb/B_oft**2))*oneovernt
    dB_phi_ds=(dBpdR*dR_ds+dBpdZ*dZ_ds)*R+Bp*dR_ds
    dB_theta_ds=dq_ds*(C_norm-B_phi)+q*(dC_norm_ds-dB_phi_ds)
    pprime=-(dB_phi_ds+aiota*dB_theta_ds)/(2.d0*twopi*sqrtg00)
!
! Fourier amplitudes of R, Z, lambda and B:
    Rmn_c(0)=real(sum(R_oft*calEm_times_Jb))*oneovernt
    Rmn_s(0)=0.d0
    Zmn_c(0)=real(sum(Z_oft*calEm_times_Jb))*oneovernt
    Zmn_s(0)=0.d0
    almn_c(0)=real(sum(al_oft*calEm_times_Jb))*oneovernt
    almn_s(0)=0.d0
    Bmn_c(0)=real(sum(B_oft*calEm_times_Jb))*oneovernt
    Bmn_s(0)=0.d0
!
    do m=1,mpol
      calEm_times_Jb=calEm_times_Jb*calE
!
      four_ampl=sum(R_oft*calEm_times_Jb)*twoovernt
      Rmn_c(m)=real(four_ampl)
      Rmn_s(m)=dimag(four_ampl)
      four_ampl=sum(Z_oft*calEm_times_Jb)*twoovernt
      Zmn_c(m)=real(four_ampl)
      Zmn_s(m)=dimag(four_ampl)
      four_ampl=sum(al_oft*calEm_times_Jb)*twoovernt
      almn_c(m)=real(four_ampl)
      almn_s(m)=dimag(four_ampl)
      four_ampl=sum(B_oft*calEm_times_Jb)*twoovernt
      Bmn_c(m)=real(four_ampl)
      Bmn_s(m)=dimag(four_ampl)
    enddo
!
! test Fourier expansion
!do it=1,nt
!  theta=htheta*dfloat(it)
!  bmod=0.d0
!  G=0.d0
!  do m=0,mpol
!    bmod=bmod+almn_c(m)*cos(m*theta)+almn_s(m)*sin(m*theta)
!    G=G+almn_c(m)*cos(m*(theta-aiota*al_oft(it)))+almn_s(m)*sin(m*(theta-aiota*al_oft(it)))
!  enddo
!  write(1001,*) theta,al_oft(it),G,bmod   !columns 2 and 3 should fit, column 4 - series over symmetry angle with Boozer amplitudes
!enddo
!close(1001)
!pause
! end test Fourier expansion
!
    write(iunit,*) '        s               iota           Jpol/nper          Itor            pprime         sqrt g(0,0)'
    write(iunit,*) '                                             [A]           [A]             [Pa]         (dV/ds)/nper'
    write(iunit,'(6e17.8)') s, aiota, -B_phi*5.d0, -sigma*B_theta*5.d0, pprime*1.d-1,              &
                                                      -sigma*sqrtg00*psitor_max*1d-6*twopi**2
    write(iunit,*) '    m    n      rmnc [m]         rmns [m]         zmnc [m]         zmns [m]'   &
                   //'         vmnc [ ]         vmns [ ]         bmnc [T]         bmns [T]'
    do m=0,mpol
      write(iunit,'(2i5,8e17.8)') m,0,Rmn_c(m)*1d-2, Rmn_s(m)*1d-2, Zmn_c(m)*1d-2, Zmn_s(m)*1d-2,  &
                     almn_c(m), almn_s(m), Bmn_c(m)*1d-4, Bmn_s(m)*1d-4
    enddo

  enddo
!
  close(iunit)
!
!
!
!
!
  open(1,form='formatted',file='box_size_axis.dat')
  write (1,*) rmn,rmx, '<= rmn, rmx (cm)'
  write (1,*) zmn,zmx, '<= zmn, zmx (cm)'
  write (1,*) raxis,zaxis, '<= raxis, zaxis (cm)'
  close(1)
!
  open(1,form='formatted',file='flux_functions.dat')
  write (1,*) '# R_beg, r,  q, psi_pol, psi_tor_vac, psi_tor'
  do i=1,nlabel
    write (1,*) rbeg(i),rsmall(i),qsaf(i),psi_pol(i),psi_tor_vac(i),psi_tor(i)*psitor_max
  enddo
  close(1)
!
  deallocate(rbeg,rsmall,qsaf,psi_pol,psi_tor_vac,psi_tor,C_const)
!
  end program efit_to_boozer
