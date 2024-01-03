! @andremfreitas dec 2023 - original code F Fossella

program Sabra   
  use Precision
  use Fixed_Parameters
  use Integration
  implicit none

  !!!!!!!!!!!!!!!!!!!!!!!!
  complex*16:: u(0:Num-1)
  real(dp):: old_input, new_input, old_flux, new_flux, old_dissipated, new_dissipated
  real(dp):: old_S1(0:Num-1), new_S1(0:Num-1), old_S2(0:Num-1), new_S2(0:Num-1)   !!structure I & II functions
  real(dp):: old_S3(0:Num-1), new_S3(0:Num-1), old_S4(0:Num-1), new_S4(0:Num-1)   !!structure III & IV functions
  real(dp):: old_S5(0:Num-1), new_S5(0:Num-1), old_S6(0:Num-1), new_S6(0:Num-1)   !!structure III & IV functions
  real(dp):: old_flusso(0:Num-1), new_flusso(0:Num-1)



  !!! write output files !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  open(1, file = 'time_velocity.csv')
  open(2, file = 'time__average_input_flux_dissipated.csv') ! for the whole system
  open(3, file = 'kn_S1_6.csv')
  open(4, file = 'time_physical.csv')
  ! open(5, file = 'n_physical.csv')


  !!!!!!!!!! Initialize forcing and k space !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  forcing=0
  do n=0,Num-1 
   k(n)=k0*(lambda**n)
   ek(n)=exp(-nu*dt*k(n)*k(n)/2)  !! ek**2=exp(-nu*t*(k**2)) is the integrating factor
   if (n==0) then 
     forcing(n)=sqrt(2.0_dp)*(dcos(pi/4)+img*dsin(pi/4)) !! Forcing acts on the first shell  <-----
   end if 
  end do
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 


  !!initializing random generator for having always the same initial condition on the velocity
  call random_init(repeatable=.true., image_distinct=.true.) 
  !! Initialize random velocity 
  t=0
  u=0
  do n=0,5   !!only the first 6 shells are not zero
    call random_number(ran)
    u(n)=0.01*k(n)**(-.33_dp)*(dcos(2*pi*ran)+img*dsin(2*pi*ran))
  end do

  !! Initial measurements
  t=t+dt
  call RK4(u)
  call physical_quantities(u,old_input,old_flux,old_dissipated) 
  call structure(u,old_S1,old_S2,old_S3,old_S4,old_S5,old_S6)
  old_flusso=(u*conjg(G(u))+conjg(u)*G(u))/2
 
  !!Evolve system and write average values
  do i=1,int(Tmax/dt) !! so integrated the model for Tmax/dt times
    t=t+dt
    call RK4(u)
    if (mod(i,100).eq.0) then
      write(1,*) t,dreal(u(4)),dreal(u(9)), dreal(u(14)) 
      write(4,*) t, sum(dreal(u*conjg(forcing)+conjg(u)*forcing))/2, sum(u*conjg(G(u))+conjg(u)*G(u))/2, &
       sum(nu*(k**2)*(dreal(u*conjg(u)))) ! input, flux, dissipation
    end if

  if (mod(i,int(measure_step/dt)).eq.0) then  !!Measurements
      call physical_quantities(u,new_input,new_flux,new_dissipated)  
      call runaverage(old_input,new_input,int((i+1)*dt/measure_step))   !! Evolve the averages
      call runaverage(old_flux,new_flux,int((i+1)*dt/measure_step))
      call runaverage(old_dissipated,new_dissipated,int((i+1)*dt/measure_step))
      
      write(2,*) t,new_input,new_flux,new_dissipated

      old_input=new_input
      old_flux=new_flux
      old_dissipated=new_dissipated

      if (i>nint(transient_for_stationarity/dt)) then
        call structure(u,new_S1,new_S2,new_S3,new_S4,new_S5,new_S6)
        new_flusso=(u*conjg(G(u))+conjg(u)*G(u))/2
        do n=0,Num-1
          call runaverage(old_S1(n),new_S1(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_S2(n),new_S2(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_S3(n),new_S3(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_S4(n),new_S4(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_S5(n),new_S5(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_S6(n),new_S6(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
          call runaverage(old_flusso(n),new_flusso(n),int(1+(i-int(transient_for_stationarity/dt))*dt/measure_step))
        end do
        old_S1=new_S1
        old_S2=new_S2
        old_S3=new_S3
        old_S4=new_S4
        old_S5=new_S5
        old_S6=new_S6
        old_flusso=new_flusso
      end if
    end if 
  end do

  !! write strcture function
  do n=0,Num-1
    write(3,*) k(n),new_S1(n),new_S2(n),new_S3(n),new_S4(n),new_S5(n),new_S6(n),new_flusso(n)
    ! write(5,*) k(n), new_input(0:Num-1)
  end do

  close(1)
  close(2)
  close(3)
  close(4)

  write(*,*) ""
end program Sabra
