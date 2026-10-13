! =============================================================================
! apple-bottom Fortran BLAS binding probe
! =============================================================================
!
! A plain CALL DGEMM / CALL ZGEMM, as in QE or Yambo, linked against
! libapplebottom. Run with AB_MODE=gpu. Like QE, the probe never calls
! ab_init(): the first call must create the Metal context itself (H-P1-2).
! Each call must reach apple-bottom's GPU path: ab_get_last_dispatch_path()
! is "gpu" and the matching ab_get_stats() GEMM counter advanced by one. A
! call that binds to Accelerate instead leaves both untouched (H-P0-1), and
! one that falls back to cblas reports "cpu". The result must also
! match a reference loop within the BLAS-ABI harness bound
!   |C - Cref| <= 1e-11 * (|alpha| (|op(A)| |op(B)|) + |beta| |C0|)
! elementwise. Leading-dimension padding is NaN, so reading it shows up.
!
! Both calls use the QE cegterg pattern: op(A) = A**T or A**H with M /= K.
! Exits with status 1 on failure.
! =============================================================================
program test_fortran_blas
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_is_nan
   implicit none

   ! ABStats in apple_bottom.h
   type, bind(C) :: ab_stats_t
      real(c_double) :: upload_time_ms, download_time_ms, kernel_time_ms
      integer(c_int64_t) :: dgemm_count, zgemm_count, elements_converted
   end type ab_stats_t

   interface
      type(ab_stats_t) function ab_get_stats() bind(C, name='ab_get_stats')
         import :: ab_stats_t
      end function ab_get_stats
      type(c_ptr) function ab_get_last_dispatch_path() &
            bind(C, name='ab_get_last_dispatch_path')
         import :: c_ptr
      end function ab_get_last_dispatch_path
   end interface

   integer, parameter :: m = 96, n = 80, k = 72, pad = 3
   integer, parameter :: lda = k + pad, ldb = k + pad, ldc = m + pad
   real(c_double), parameter :: tol = 1.0d-11
   integer :: failures = 0

   call check_dgemm()
   call check_zgemm()

   if (failures > 0) then
      print '(i0,a)', failures, ' Fortran BLAS check(s) failed'
      stop 1
   end if
   print '(a)', 'Fortran BLAS binding probe: all checks passed'

contains

   ! C = alpha * A**T * B + beta * C, A stored k x m.
   subroutine check_dgemm()
      real(c_double), allocatable :: a(:, :), b(:, :), c(:, :), c0(:, :)
      real(c_double), parameter :: alpha = 0.75d0, beta = -0.5d0
      real(c_double) :: nan, s, bound, e, err
      type(ab_stats_t) :: st0, st1
      integer :: i, j, l

      nan = ieee_value(1.0d0, ieee_quiet_nan)
      allocate(a(lda, m), b(ldb, n), c0(ldc, n), source=nan)
      do j = 1, m
         do i = 1, k
            a(i, j) = sin(0.37d0 * i + 1.71d0 * j)
         end do
      end do
      do j = 1, n
         do i = 1, k
            b(i, j) = cos(0.53d0 * i - 0.29d0 * j)
         end do
         do i = 1, m
            c0(i, j) = sin(0.11d0 * i * j)
         end do
      end do
      c = c0

      st0 = ab_get_stats()
      call dgemm('T', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      st1 = ab_get_stats()
      call check_route('DGEMM', st1%dgemm_count - st0%dgemm_count)

      err = 0
      do j = 1, n
         do i = 1, m
            s = 0
            bound = 0
            do l = 1, k
               s = s + a(l, i) * b(l, j)
               bound = bound + abs(a(l, i) * b(l, j))
            end do
            bound = abs(alpha) * bound + abs(beta * c0(i, j))
            e = abs(c(i, j) - (alpha * s + beta * c0(i, j))) / bound
            if (ieee_is_nan(e) .or. e > err) err = e     ! a NaN sticks
         end do
      end do
      call check_error('DGEMM', err)
   end subroutine check_dgemm

   ! C = alpha * A**H * B + beta * C, A stored k x m.
   subroutine check_zgemm()
      complex(c_double_complex), allocatable :: a(:, :), b(:, :), c(:, :), c0(:, :)
      complex(c_double_complex), parameter :: alpha = (0.5d0, 0.25d0)
      complex(c_double_complex), parameter :: beta = (-0.5d0, 0.125d0)
      complex(c_double_complex) :: nan, s
      real(c_double) :: bound, e, err
      type(ab_stats_t) :: st0, st1
      integer :: i, j, l

      nan = cmplx(ieee_value(1.0d0, ieee_quiet_nan), 0, kind=c_double)
      allocate(a(lda, m), b(ldb, n), c0(ldc, n), source=nan)
      do j = 1, m
         do i = 1, k
            a(i, j) = cmplx(sin(0.37d0 * i + 1.71d0 * j), &
                            cos(0.19d0 * i - 0.83d0 * j), kind=c_double)
         end do
      end do
      do j = 1, n
         do i = 1, k
            b(i, j) = cmplx(cos(0.53d0 * i - 0.29d0 * j), &
                            sin(0.41d0 * i + 0.67d0 * j), kind=c_double)
         end do
         do i = 1, m
            c0(i, j) = cmplx(sin(0.11d0 * i * j), cos(0.07d0 * (i + j)), kind=c_double)
         end do
      end do
      c = c0

      st0 = ab_get_stats()
      call zgemm('C', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      st1 = ab_get_stats()
      call check_route('ZGEMM', st1%zgemm_count - st0%zgemm_count)

      err = 0
      do j = 1, n
         do i = 1, m
            s = 0
            bound = 0
            do l = 1, k
               s = s + conjg(a(l, i)) * b(l, j)
               bound = bound + abs(a(l, i)) * abs(b(l, j))
            end do
            bound = abs(alpha) * bound + abs(beta) * abs(c0(i, j))
            e = abs(c(i, j) - (alpha * s + beta * c0(i, j))) / bound
            if (ieee_is_nan(e) .or. e > err) err = e
         end do
      end do
      call check_error('ZGEMM', err)
   end subroutine check_zgemm

   subroutine check_route(name, gpu_calls)
      character(*), intent(in) :: name
      integer(c_int64_t), intent(in) :: gpu_calls
      character(kind=c_char), pointer :: p(:)
      character(4) :: path
      integer :: i

      ! "cpu", "gpu" or "none", NUL-terminated
      call c_f_pointer(ab_get_last_dispatch_path(), p, [4])
      path = ''
      do i = 1, 4
         if (p(i) == c_null_char) exit
         path(i:i) = p(i)
      end do

      if (path == 'gpu' .and. gpu_calls == 1) then
         print '(a)', 'PASS ' // name // ' reached the apple-bottom GPU path'
      else
         print '(a,i0,a)', 'FAIL ' // name // ' did not reach apple-bottom: last path "' // &
            trim(path) // '", GPU GEMMs ', gpu_calls, ' (expected "gpu", 1)'
         failures = failures + 1
      end if
   end subroutine check_route

   subroutine check_error(name, err)
      character(*), intent(in) :: name
      real(c_double), intent(in) :: err

      if (err <= tol) then
         print '(a,es9.2)', 'PASS ' // name // ' result, max scaled error', err
      else
         print '(a,es9.2,a,es9.2)', 'FAIL ' // name // ' result, max scaled error', err, &
            ' > ', tol
         failures = failures + 1
      end if
   end subroutine check_error

end program test_fortran_blas
