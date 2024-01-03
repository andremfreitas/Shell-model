! This module sets a public, hardcoded (this is what parameter means in fortran) integer, called 'dp', the function selected_real_kind(8). 
! is used to specify the precision of real numbers, and in this case, it's set to 8bytes (64 bits aka float64).
module Precision
  integer, parameter, public :: dp = selected_real_kind(8)
end module Precision
  
  
! This module contains fixed public parameters available to all subroutine, functions, etc.
module Fixed_Parameters
  use Precision 
  implicit none

  ! Fixed public parameters
  integer, parameter, public:: Num=20                ! Number of shells
  real(dp), parameter, public:: dt=1.e-5, nu=1.e-6, Tmax=500, transient_for_stationarity=20, measure_step=1.e-3
  real(dp), parameter, public:: lambda=2, k0=1, a=1, b=-0.5, c=-0.5, pi=3.14159265 
  complex*16, public, parameter:: img=(0.0_dp,1.0_dp)
  complex*16, public:: forcing(0:Num-1)              ! Array for forcing
  real(dp), public:: t, ran, k(0:Num-1), ek(0:Num-1) 
  integer, public :: n,i                             ! n is Shell index,dummy indices

end module Fixed_Parameters


module Integration
    use Precision
    use Fixed_Parameters
    implicit none

  contains

    ! Non linear coupling G[u,u] 
    function G(u)      
      complex*16,intent(in):: u(0:Num-1)
      complex*16:: G(0:Num-1) 

      do n=0,Num-1
        if (n==0) then 
          G(n)=img*(a*k(n+1)*conjg(u(n+1))*u(n+2))
        else if (n==1) then 
          G(n)=img*(a*k(n+1)*conjg(u(n+1))*u(n+2)+b*k(n)*conjg(u(n-1))*u(n+1))
        else if (n==Num-2) then 
          G(n)=img*(b*k(n)*conjg(u(n-1))*u(n+1)-c*k(n-1)*u(n-1)*u(n-2))
        else if(n==Num-1) then 
          G(n)=img*(-c*k(n-1)*u(n-1)*u(n-2))
        else 
          G(n)=img*(a*k(n+1)*conjg(u(n+1))*u(n+2)+b*k(n)*conjg(u(n-1))*u(n+1)-c*k(n-1)*u(n-1)*u(n-2))
        end if 
      end do
    end function G



  ! This subroutine integrates with RK4 scheme the Sabra shell model for turbulence
  ! The solution is simplified through the integrating factor method 
    subroutine RK4(u)       
      implicit none 
      complex*16,intent(inout):: u(0:Num-1)                 ! Velocity on the shells -- inout allows read and write
      complex*16,allocatable:: A1(:),A2(:),A3(:),A4(:)      ! RungeKutta increments
      
      ! Allocating RK increments 
      allocate(A1(0:Num-1),A2(0:Num-1),A3(0:Num-1),A4(0:Num-1))
  
      A1=dt*(forcing+G(u))    !! increments
      A2=dt*(forcing+G(ek*(u+A1/2)))
      A3=dt*(forcing+G(ek*u+A2/2))
      A4=dt*(forcing+G(u*(ek**2)+ek*A3))

      u=(ek**2)*(u+A1/6)+ek*(A2+A3)/3+A4/6
      
    deallocate(A1,A2,A3,A4)
    end subroutine Rk4



    ! This routine returns some physical quantities of interest
    subroutine physical_quantities(u,input_energy,flux_energy,dissipated_energy)
      complex*16, intent(in):: u(0:Num-1)
      real(dp), intent(out):: input_energy, flux_energy, dissipated_energy

      input_energy=sum(dreal(u*conjg(forcing)+conjg(u)*forcing))/2
      flux_energy=sum(u*conjg(G(u))+conjg(u)*G(u))/2
      dissipated_energy=sum(nu*(k**2)*(dreal(u*conjg(u))))   
    end subroutine physical_quantities

    !! This routine returns some physical quantities of interest
    subroutine physical_quantities2(u,input_energy,flux_energy,dissipated_energy)
      complex*16, intent(in):: u(0:Num-1)
      real(dp), intent(out):: input_energy(0:Num-1), flux_energy(0:Num-1), dissipated_energy(0:Num-1)
      real(dp), allocatable:: suppI(:), suppF(:), suppD(:)
      
      allocate(suppI(0:Num-1),suppF(0:Num-1),suppD(0:Num-1))
      suppI=dreal(u*conjg(forcing))              ! Input energy at the n-th scale
      suppF=dreal(u*conjg(G(u)))                 ! Energy flux at the n-th scale
      suppD=nu*(k**2)*(dreal(u*conjg(u)))        ! Dissipated energy at the n-th scale

      do n=0,Num-1
          input_energy=sum(suppI(0:n))           ! Input energy up to the n-th scale
          flux_energy(n)=sum(suppF(0:n))         ! Energy flux up to the n-th scale
          dissipated_energy(n)=sum(suppD(0:n))   ! Dissipated energy up to the n-th scale
      end do            
      deallocate(suppI,suppF,suppD)
  end subroutine physical_quantities2
    
    ! Returns structure function 
    subroutine structure(u,S1,S2,S3,S4,S5,S6)
      complex*16, intent(in):: u(0:Num-1)
      real(dp), intent(out):: S1(0:Num-1),S2(0:Num-1),S3(0:Num-1)
      real(dp), intent(out):: S4(0:Num-1),S5(0:Num-1),S6(0:Num-1)

      S1=sqrt(dreal(u*conjg(u)))
      S2=sqrt(dreal(u*conjg(u)))**2
      S3=sqrt(dreal(u*conjg(u)))**3
      S4=sqrt(dreal(u*conjg(u)))**4
      S5=sqrt(dreal(u*conjg(u)))**5
      S6=sqrt(dreal(u*conjg(u)))**6
    end subroutine structure 


    ! Returns the running average 
    subroutine runaverage(old,new,i)
      integer, intent(in):: i
      real(dp), intent(in):: old
      real(dp), intent(inout):: new

      !new=new/(i+2)+(i+1)*old/(i+2)
      new=old+(new-old)/i
    end subroutine runaverage 



  end module Integration
  
