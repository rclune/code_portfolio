subroutine aabb_to_aabb( &
    Cstart, &
    Cend, &
    Vstart, &
    Vend, &
    Hstart, &
    Hend, &
    tbm, &
    tbm11, &
    tbm13, &
    tbm15, &
    tbm17, &
    tbm19, &
    tbm21, &
    tbm23, &
    tbm25, &
    tbm27, &
    tbm29, &
    tbm3, &
    tbm31, &
    tbm33, &
    tbm35, &
    tbm37, &
    tbm39, &
    tbm5, &
    tbm7, &
    tbm9, &
    C0_coeff, &
    C0_coeff_2, &
    Ca_coeff_b, &
    Ca_coeff_k, &
    Cb_coeff_b, &
    Cb_coeff_k, &
    f_oo, &
    f_ov, &
    f_vo, &
    f_vv, &
    t4, &
    tf_oo, &
    tf_oo2, &
    tf_oo3, &
    tf_oo4, &
    tf_oo5, &
    tf_oo6, &
    tf_oo7, &
    tf_oo8, &
    tf_oo9, &
    to_4)

implicit none

integer, intent(in) :: Cstart
integer, intent(in) :: Cend
integer, intent(in) :: Vstart
integer, intent(in) :: Vend
integer, intent(in) :: Hstart
integer, intent(in) :: Hend
real(kind=8), intent(inout) :: tbm(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm11(Cstart:Cend,Cstart:Cend,Vstart:Vend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm13(Cstart:Cend,Cstart:Cend,Vstart:Vend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm15(Cstart:Cend,Vstart:Vend,Vstart:Vend,Cstart:Cend)
real(kind=8), intent(inout) :: tbm17(Cstart:Cend,Vstart:Vend,Vstart:Vend,Cstart:Cend)
real(kind=8), intent(inout) :: tbm19(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm21(Vstart:Vend,Cstart:Cend,Cstart:Cend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm23(Cstart:Cend,Cstart:Cend,Vstart:Vend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm25(Cstart:Cend,Cstart:Cend)
real(kind=8), intent(inout) :: tbm27(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm29(Vstart:Vend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm3(Vstart:Vend,Cstart:Cend)
real(kind=8), intent(inout) :: tbm31
real(kind=8), intent(inout) :: tbm33
real(kind=8), intent(inout) :: tbm35
real(kind=8), intent(inout) :: tbm37
real(kind=8), intent(inout) :: tbm39(Vstart:Vend,Cstart:Cend)
real(kind=8), intent(inout) :: tbm5(Cstart:Cend,Vstart:Vend,Vstart:Vend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm7(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(inout) :: tbm9(Cstart:Cend,Cstart:Cend,Vstart:Vend,Cstart:Cend)
real(kind=8), intent(in) :: C0_coeff
real(kind=8), intent(in) :: C0_coeff_2
real(kind=8), intent(in) :: Ca_coeff_b(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(in) :: Ca_coeff_k(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(in) :: Cb_coeff_b(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(in) :: Cb_coeff_k(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(in) :: f_oo(Cstart:Cend,Cstart:Cend)
real(kind=8), intent(in) :: f_ov(Cstart:Cend,Vstart:Vend)
real(kind=8), intent(in) :: f_vo(Vstart:Vend,Cstart:Cend)
real(kind=8), intent(in) :: f_vv(Vstart:Vend,Vstart:Vend)
real(kind=8), intent(in) :: t4(Cstart:Cend,Cstart:Cend,Vstart:Vend,Vstart:Vend)
real(kind=8), intent(in) :: tf_oo
real(kind=8), intent(in) :: tf_oo2
real(kind=8), intent(in) :: tf_oo3
real(kind=8), intent(in) :: tf_oo4
real(kind=8), intent(in) :: tf_oo5
real(kind=8), intent(in) :: tf_oo6
real(kind=8), intent(in) :: tf_oo7
real(kind=8), intent(in) :: tf_oo8
real(kind=8), intent(in) :: tf_oo9
real(kind=8), intent(inout) :: to_4(Cstart:Cend,Cstart:Cend,Vstart:Vend,Vstart:Vend)

integer :: i
integer :: j
integer :: ka
integer :: mb
integer :: a
integer :: b
integer :: ca
integer :: eb
real(kind=8), allocatable :: temp_ten_0(:,:,:,:)
real(kind=8), allocatable :: temp_ten_1(:,:)
real(kind=8), allocatable :: temp_ten_2(:,:)
real(kind=8), allocatable :: temp_ten_3(:,:)
real(kind=8), allocatable :: temp_ten_4(:,:,:,:)
real(kind=8) :: temp_ten_5


allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do b = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_0(mb,b,a,i) = t4(i,mb,a,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_1(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_1)

allocate(temp_ten_1(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_1(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_1, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_1)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm(mb,eb) = (0.00d+00) * tbm(mb,eb) + (1.00d+00) * temp_ten_3(mb,eb)
enddo
enddo
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) &
  * Ca_coeff_k(ka,ca) &
  * tbm(mb,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do j = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_0(j,eb,a,i) = t4(i,j,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(eb,j) = temp_ten_2(j,eb)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_2(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm3(eb,mb) = (0.00d+00) * tbm3(eb,mb) + (-1.00d+00) * temp_ten_1(eb,mb)
enddo
enddo
deallocate(temp_ten_1)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) &
  * Ca_coeff_k(ka,ca) &
  * tbm3(eb,mb)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_3(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = temp_ten_2(j,a)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do j = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_0(a,j,mb,eb) = t4(j,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_0)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_2(mb,eb) &
  * Ca_coeff_k(ka,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do i = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_0(b,i,mb,eb) = t4(i,mb,b,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           1, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           1 &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_0)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_3(mb,eb) &
  * Ca_coeff_k(ka,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_3)

allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_0(mb,eb,a,i) = t4(i,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (2.00d+00*tf_oo) * temp_ten_2(mb,eb) &
  * Ca_coeff_k(ka,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
do b = Vstart, Vend
temp_ten_4(ka,mb,eb,b) = temp_ten_0(ka,mb,b,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(ka,mb,eb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_0(ka,ca,eb,j) = temp_ten_4(ka,j,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_3(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_4(ka,ca,eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(mb,ca,eb,i) = t4(i,mb,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do a = Vstart, Vend
do ca = Vstart, Vend
do eb = Vstart, Vend
do mb = Cstart, Cend
tbm5(mb,ca,eb,a) = (0.00d+00) * tbm5(mb,ca,eb,a) + (-2.00d+00) * temp_ten_0(mb,ca,eb,a)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do a = Vstart, Vend
do mb = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(a,mb,ca,eb) = tbm5(mb,ca,eb,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*tf_oo2) * temp_ten_0(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(mb,eb,a,i) = t4(i,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm7(mb,eb) = (0.00d+00) * tbm7(mb,eb) + (1.00d+00) * temp_ten_2(mb,eb)
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_2(ka,ca) &
  * tbm7(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_1)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do b = Vstart, Vend
temp_ten_4(a,ka,mb,b) = t4(ka,mb,a,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_4(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do j = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,j,mb,eb) = t4(j,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do mb = Cstart, Cend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(ca,mb,eb,j) = temp_ten_0(ca,j,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_3(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(ca,mb,eb,ka)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do j = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,j,eb) = t4(ka,j,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do ka = Cstart, Cend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(ca,ka,eb,j) = temp_ten_0(ca,ka,j,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_3(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(ca,ka,eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(b,ka,mb,eb) = t4(ka,mb,b,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
temp_ten_4(ka,mb,eb,a) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do eb = Vstart, Vend
do i = Cstart, Cend
do ka = Cstart, Cend
do mb = Cstart, Cend
tbm9(ka,mb,eb,i) = (0.00d+00) * tbm9(ka,mb,eb,i) + (-2.00d+00) * temp_ten_0(ka,mb,eb,i)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(i,ka,mb,eb) = tbm9(ka,mb,eb,i)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*tf_oo3) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
do b = Vstart, Vend
temp_ten_4(ka,mb,eb,b) = t4(ka,mb,b,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
tbm11(ka,mb,eb,ca) = (0.00d+00) * tbm11(ka,mb,eb,ca) + (1.00d+00) * temp_ten_0(ka,mb,eb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * tbm11(ka,mb,eb,ca)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
tbm13(ka,mb,ca,eb) = (0.00d+00) * tbm13(ka,mb,ca,eb) + (1.00d+00) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * tbm13(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(mb,ca,eb,j) = t4(j,mb,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_3(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
tbm15(mb,ca,eb,ka) = (0.00d+00) * tbm15(mb,ca,eb,ka) + (-1.00d+00) * temp_ten_0(mb,ca,eb,ka)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * tbm15(mb,ca,eb,ka)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(ka,ca,eb,j) = t4(ka,j,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_3(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
tbm17(ka,ca,eb,mb) = (0.00d+00) * tbm17(ka,ca,eb,mb) + (-1.00d+00) * temp_ten_0(ka,ca,eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * tbm17(ka,ca,eb,mb)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (2.00d+00*tf_oo4) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = temp_ten_2(i,ca)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,mb,eb) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = temp_ten_2(i,a)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(mb,eb,a,i) = t4(i,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm19(mb,eb) = (0.00d+00) * tbm19(mb,eb) + (-1.00d+00) * temp_ten_2(mb,eb)
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(ca,j) = Ca_coeff_k(j,ca)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_2(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_1(ca,ka) &
  * tbm19(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_1)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(ca,j) = Ca_coeff_k(j,ca)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_2(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1), &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_1)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,mb,eb) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = Ca_coeff_k(j,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_2(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Cstart:Cend, &
                    Vstart:Vend))
do ka = Cstart, Cend
do a = Vstart, Vend
temp_ten_3(ka,a) = temp_ten_1(a,ka)
enddo
enddo
deallocate(temp_ten_1)
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           temp_ten_2, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_1)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = Ca_coeff_k(j,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_2(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           1, &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ka = Cstart, Cend
do i = Cstart, Cend
do a = Vstart, Vend
temp_ten_4(b,ka,i,a) = t4(ka,i,b,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do a = Vstart, Vend
do ca = Vstart, Vend
do i = Cstart, Cend
do ka = Cstart, Cend
tbm21(ca,ka,i,a) = (0.00d+00) * tbm21(ca,ka,i,a) + (1.00d+00) * temp_ten_0(ca,ka,i,a)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
do ca = Vstart, Vend
do ka = Cstart, Cend
temp_ten_4(a,i,ca,ka) = tbm21(ca,ka,i,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           1, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_2(ca,ka) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )

do a = Vstart, Vend
do ca = Vstart, Vend
do i = Cstart, Cend
do ka = Cstart, Cend
tbm23(ka,i,ca,a) = (0.00d+00) * tbm23(ka,i,ca,a) + (-1.00d+00) * temp_ten_4(ka,i,ca,a)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,i,ka,ca) = tbm23(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_2(ka,ca) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_3(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = temp_ten_2(j,a)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do j = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,j,ka,ca) = t4(ka,j,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_2(ka,ca) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(b,i,ka,ca) = t4(ka,i,ca,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           1, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           1 &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_3(ka,ca) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,a,i) = t4(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (2.00d+00*tf_oo5) * temp_ten_2(ka,ca) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,b,eb) = t4(ka,i,b,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do mb = Cstart, Cend
do ka = Cstart, Cend
do eb = Vstart, Vend
do b = Vstart, Vend
temp_ten_4(mb,ka,eb,b) = temp_ten_0(mb,ka,b,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(mb,ka,eb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do b = Vstart, Vend
temp_ten_4(i,ka,ca,b) = t4(ka,i,ca,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_4(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do j = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,j,ca,eb) = t4(j,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(mb,ca,eb,j) = temp_ten_0(mb,j,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_3(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(mb,ca,eb,ka)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do j = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(j,ka,ca,eb) = t4(ka,j,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do i = Cstart, Cend
do mb = Cstart, Cend
tbm25(mb,i) = (0.00d+00) * tbm25(mb,i) + (-2.00d+00) * temp_ten_2(mb,i)
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,eb,i) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do i = Cstart, Cend
do mb = Cstart, Cend
temp_ten_3(i,mb) = tbm25(mb,i)
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*tf_oo6) * temp_ten_0(ka,ca,eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm27(mb,eb) = (0.00d+00) * tbm27(mb,eb) + (1.00d+00) * temp_ten_2(mb,eb)
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,a,i) = t4(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_2(ka,ca) &
  * tbm27(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,ca,eb) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do b = Vstart, Vend
temp_ten_4(a,ka,mb,b) = t4(ka,mb,b,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ca = Vstart, Vend
temp_ten_3(b,ca) = f_vv(ca,b)
enddo
enddo
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_0)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_4(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do j = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,j,mb,ca) = t4(j,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do mb = Cstart, Cend
do ca = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(eb,mb,ca,j) = temp_ten_0(eb,j,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do ka = Cstart, Cend
temp_ten_3(j,ka) = f_oo(ka,j)
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(eb,mb,ca,ka)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do j = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,j,ca) = t4(ka,j,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do ka = Cstart, Cend
do ca = Vstart, Vend
do j = Cstart, Cend
temp_ten_4(eb,ka,ca,j) = temp_ten_0(eb,ka,j,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)
allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_3(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(eb,ka,ca,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(b,ka,mb,ca) = t4(ka,mb,ca,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

do a = Vstart, Vend
do eb = Vstart, Vend
tbm29(eb,a) = (0.00d+00) * tbm29(eb,a) + (-2.00d+00) * temp_ten_2(eb,a)
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do a = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(a,eb) = tbm29(eb,a)
enddo
enddo
allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*tf_oo7) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

tbm31 = (0.00d+00) * tbm31 + (1.00d+00) * temp_ten_5

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(b,ka,mb,eb) = t4(ka,mb,b,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (tbm31) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

tbm33 = (0.00d+00) * tbm33 + (1.00d+00) * temp_ten_5

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do b = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(b,ka,mb,ca) = t4(ka,mb,ca,b)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (tbm33) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

tbm35 = (0.00d+00) * tbm35 + (-1.00d+00) * temp_ten_5

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (tbm35) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

tbm37 = (0.00d+00) * tbm37 + (-1.00d+00) * temp_ten_5

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do j = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(j,ka,ca,eb) = t4(ka,j,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (tbm37) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (2.00d+00*tf_oo8) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do eb = Vstart, Vend
temp_ten_3(b,eb) = f_vv(eb,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = temp_ten_2(i,eb)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,mb,ca) = t4(ka,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do b = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(b,a) = f_vv(a,b)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = temp_ten_2(i,a)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(eb,j) = Cb_coeff_k(j,eb)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_2(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do eb = Vstart, Vend
do mb = Cstart, Cend
tbm39(eb,mb) = (0.00d+00) * tbm39(eb,mb) + (-1.00d+00) * temp_ten_1(eb,mb)
enddo
enddo
deallocate(temp_ten_1)

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,a,i) = t4(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_2(ka,ca) &
  * tbm39(eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(eb,j) = Cb_coeff_k(j,eb)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_2(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_1)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,mb,ca) = t4(ka,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = Cb_coeff_k(j,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do mb = Cstart, Cend
temp_ten_2(j,mb) = f_oo(mb,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Cstart:Cend, &
                    Vstart:Vend))
do mb = Cstart, Cend
do a = Vstart, Vend
temp_ten_3(mb,a) = temp_ten_1(a,mb)
enddo
enddo
deallocate(temp_ten_1)
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_1(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           temp_ten_2, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,ca,eb) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do j = Cstart, Cend
temp_ten_3(a,j) = Cb_coeff_k(j,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
do j = Cstart, Cend
do i = Cstart, Cend
temp_ten_2(j,i) = f_oo(i,j)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_1, &
           1, &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_1)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,mb,eb) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff_2) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,mb,ca) = t4(ka,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_vv( &
           Vstart:Vend, &
           Vstart:Vend), &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff_2) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff_2) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,ca,eb) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_oo( &
           Cstart:Cend, &
           Cstart:Cend), &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff_2) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (2.00d+00*C0_coeff_2*tf_oo) &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(mb,eb,a,i) = t4(i,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_2(mb,eb) &
  * f_vo(ca,ka)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ca = Vstart, Vend
temp_ten_2(i,ca) = f_vo(ca,i)
enddo
enddo
allocate(temp_ten_1(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_2, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_1, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
do ca = Vstart, Vend
do a = Vstart, Vend
temp_ten_3(ca,a) = temp_ten_1(a,ca)
enddo
enddo
deallocate(temp_ten_1)
allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,mb,eb) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do ka = Cstart, Cend
temp_ten_3(a,ka) = f_ov(ka,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do i = Cstart, Cend
temp_ten_3(ka,i) = temp_ten_2(i,ka)
enddo
enddo
deallocate(temp_ten_2)
allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = f_ov(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do mb = Cstart, Cend
do eb = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(mb,eb,a,i) = t4(i,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = f_ov(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_2(mb,eb) &
  * Ca_coeff_k(ka,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Ca_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           f_vo( &
           Vstart:Vend, &
           Cstart:Cend), &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1) &
           )

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           t4( &
           Cstart:Cend, &
           Cstart:Cend, &
           Vstart:Vend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_4)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do ca = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(ca,i) = Ca_coeff_k(i,ca)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           f_ov( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do eb = Vstart, Vend
temp_ten_4(a,ka,mb,eb) = t4(ka,mb,a,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(ca,ka,mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Ca_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = f_ov(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,a,i) = t4(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_2(ka,ca) &
  * f_vo(eb,mb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           f_vo( &
           Vstart:Vend, &
           Cstart:Cend), &
           (Vend-Vstart+1), &
           Cb_coeff_b( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1) &
           )

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,mb,ca) = t4(ka,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_b(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           f_ov( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,ca,eb) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = f_ov(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = Cb_coeff_b(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

allocate(temp_ten_4(Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend, &
                    Cstart:Cend))
do ka = Cstart, Cend
do ca = Vstart, Vend
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_4(ka,ca,a,i) = t4(ka,i,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = f_ov(i,a)
enddo
enddo
allocate(temp_ten_2(Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_4, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1), &
           temp_ten_3, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Cend-Cstart+1) * &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_4)
deallocate(temp_ten_3)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_2(ka,ca) &
  * Cb_coeff_k(mb,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_2)

allocate(temp_ten_3(Cstart:Cend, &
                    Cstart:Cend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           Cb_coeff_k( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           f_vo( &
           Vstart:Vend, &
           Cstart:Cend), &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1) &
           )

allocate(temp_ten_4(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
do i = Cstart, Cend
do ka = Cstart, Cend
do ca = Vstart, Vend
do eb = Vstart, Vend
temp_ten_4(i,ka,ca,eb) = t4(ka,i,ca,eb)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Cend-Cstart+1), &
           (Vend-Vstart+1) * &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Cend-Cstart+1), &
           temp_ten_4, &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Cend-Cstart+1) &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(mb,ka,ca,eb)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do eb = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(eb,i) = Cb_coeff_k(i,eb)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1), &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           (Vend-Vstart+1), &
           f_ov( &
           Cstart:Cend, &
           Vstart:Vend), &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_3)

allocate(temp_ten_4(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
do a = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
do ca = Vstart, Vend
temp_ten_4(a,ka,mb,ca) = t4(ka,mb,ca,a)
enddo
enddo
enddo
enddo
allocate(temp_ten_0(Vstart:Vend, &
                    Cstart:Cend, &
                    Cstart:Cend, &
                    Vstart:Vend))
call dgemm('N', 'N', &
           (Vend-Vstart+1), &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1) * &
           (Cend-Cstart+1), &
           (Vend-Vstart+1), &
           1.00d+00, &
           temp_ten_2, &
           (Vend-Vstart+1), &
           temp_ten_4, &
           (Vend-Vstart+1), &
           0.00d+00, &
           temp_ten_0, &
           (Vend-Vstart+1) &
           )
deallocate(temp_ten_2)
deallocate(temp_ten_4)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (-1.00d+00*C0_coeff) * temp_ten_0(eb,ka,mb,ca)
enddo
enddo
enddo
enddo
deallocate(temp_ten_0)

allocate(temp_ten_3(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_3(a,i) = Cb_coeff_k(i,a)
enddo
enddo
allocate(temp_ten_2(Vstart:Vend, &
                    Cstart:Cend))
do a = Vstart, Vend
do i = Cstart, Cend
temp_ten_2(a,i) = f_ov(i,a)
enddo
enddo
call dgemm('N', 'N', &
           1, &
           1, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           1.00d+00, &
           temp_ten_3, &
           1, &
           temp_ten_2, &
           (Vend-Vstart+1) * &
           (Cend-Cstart+1), &
           0.00d+00, &
           temp_ten_5, &
           1 &
           )
deallocate(temp_ten_3)
deallocate(temp_ten_2)

do ca = Vstart, Vend
do eb = Vstart, Vend
do ka = Cstart, Cend
do mb = Cstart, Cend
to_4(ka,mb,ca,eb) = (1.00d+00) * to_4(ka,mb,ca,eb) + (1.00d+00*C0_coeff) * temp_ten_5 &
  * t4(ka,mb,ca,eb)
enddo
enddo
enddo
enddo

end subroutine aabb_to_aabb
