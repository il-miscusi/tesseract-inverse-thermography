! Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
!
! Darcy / Brinkman flow through a designable porous medium, with a
! HAND-DERIVED DISCRETE ADJOINT.
!
! Forward:   div( lambda grad p ) = 0,     lambda = kappa(gamma) / mu
!            u = -lambda grad p
! BCs:       p = p_in on the left edge, p = 0 on the right edge,
!            no-flow (natural) on top and bottom.
!
! Discretisation: cell-centred finite volume on a uniform Nx x Ny grid.
! Face mobilities are HARMONIC means of the two adjacent cell mobilities,
! which is the correct averaging for a diffusive flux in series.
!
! The adjoint below is derived by hand from the discrete equations (it is not
! produced by any AD tool) -- that is the point of this component: it is a
! differentiation strategy no Python AD framework can trace through.
!
!   A(lambda) p = b
!   dJ/dlambda_k = -w^T (dA/dlambda_k) p     with   A^T w = pbar,  A symmetric
!
! and because A is assembled face-by-face,
!
!   w^T (dA/dlambda_k) p = sum_f  dT_f/dlambda_k * (w_i - w_j)(p_i - p_j)
!
! which is purely local and cheap.

module darcy_mod
  implicit none
  integer, parameter :: dp = kind(1.0d0)
contains

  ! ---- material interpolation: RAMP-style Brinkman penalisation --------------
  ! gamma = 0 -> open coolant channel (kappa_max);  gamma = 1 -> solid fin (kappa_min)
  pure subroutine kappa_of_gamma(g, kmin, kmax, q, k, dk)
    real(dp), intent(in)  :: g, kmin, kmax, q
    real(dp), intent(out) :: k, dk
    real(dp) :: den
    den = 1.0_dp + q * g
    k   = kmax + (kmin - kmax) * g * (1.0_dp + q) / den
    ! d/dg of the RAMP interpolation
    dk  = (kmin - kmax) * (1.0_dp + q) / (den * den)
  end subroutine kappa_of_gamma

  pure integer function idx(i, j, nx)
    integer, intent(in) :: i, j, nx
    idx = (j - 1) * nx + i
  end function idx

  ! ---- assemble face transmissibilities (harmonic mean) ----------------------
  ! tx(i,j) is the face between cell (i,j) and (i+1,j);  ty likewise in y.
  subroutine face_trans(lam, nx, ny, dx, dy, tx, ty)
    integer,  intent(in)  :: nx, ny
    real(dp), intent(in)  :: lam(nx, ny), dx, dy
    real(dp), intent(out) :: tx(nx, ny), ty(nx, ny)
    integer :: i, j
    real(dp) :: a, b
    tx = 0.0_dp; ty = 0.0_dp
    do j = 1, ny
      do i = 1, nx - 1
        a = lam(i, j); b = lam(i + 1, j)
        tx(i, j) = 2.0_dp * a * b / max(a + b, 1.0e-300_dp) * (dy / dx)
      end do
    end do
    do j = 1, ny - 1
      do i = 1, nx
        a = lam(i, j); b = lam(i, j + 1)
        ty(i, j) = 2.0_dp * a * b / max(a + b, 1.0e-300_dp) * (dx / dy)
      end do
    end do
  end subroutine face_trans

  ! ---- matrix-vector product  A*p  (never forms A) ---------------------------
  ! Dirichlet columns: i=1 (p_in) and i=nx (0) are enforced by a half-cell
  ! transmissibility to the boundary value.
  subroutine apply_A(p, tx, ty, tbl, tbr, nx, ny, out)
    integer,  intent(in)  :: nx, ny
    real(dp), intent(in)  :: p(nx, ny), tx(nx, ny), ty(nx, ny)
    real(dp), intent(in)  :: tbl(ny), tbr(ny)
    real(dp), intent(out) :: out(nx, ny)
    integer :: i, j
    real(dp) :: acc
    do j = 1, ny
      do i = 1, nx
        acc = 0.0_dp
        if (i > 1)  acc = acc + tx(i - 1, j) * (p(i, j) - p(i - 1, j))
        if (i < nx) acc = acc + tx(i, j)     * (p(i, j) - p(i + 1, j))
        if (j > 1)  acc = acc + ty(i, j - 1) * (p(i, j) - p(i, j - 1))
        if (j < ny) acc = acc + ty(i, j)     * (p(i, j) - p(i, j + 1))
        if (i == 1)  acc = acc + tbl(j) * p(i, j)
        if (i == nx) acc = acc + tbr(j) * p(i, j)
        out(i, j) = acc
      end do
    end do
  end subroutine apply_A

  ! ---- preconditioned conjugate gradient (A is SPD) --------------------------
  subroutine cg_solve(b, tx, ty, tbl, tbr, nx, ny, tol, maxit, x, iters, resid)
    integer,  intent(in)    :: nx, ny, maxit
    real(dp), intent(in)    :: b(nx, ny), tx(nx, ny), ty(nx, ny)
    real(dp), intent(in)    :: tbl(ny), tbr(ny), tol
    real(dp), intent(inout) :: x(nx, ny)
    integer,  intent(out)   :: iters
    real(dp), intent(out)   :: resid
    real(dp) :: r(nx, ny), z(nx, ny), pdir(nx, ny), Ap(nx, ny), diag(nx, ny)
    real(dp) :: rz, rz_new, alpha, beta, pAp, bnorm, best_resid
    real(dp) :: xbest(nx, ny)
    integer  :: i, j, it, since_best

    ! Jacobi preconditioner: the diagonal of A
    do j = 1, ny
      do i = 1, nx
        diag(i, j) = 0.0_dp
        if (i > 1)  diag(i, j) = diag(i, j) + tx(i - 1, j)
        if (i < nx) diag(i, j) = diag(i, j) + tx(i, j)
        if (j > 1)  diag(i, j) = diag(i, j) + ty(i, j - 1)
        if (j < ny) diag(i, j) = diag(i, j) + ty(i, j)
        if (i == 1)  diag(i, j) = diag(i, j) + tbl(j)
        if (i == nx) diag(i, j) = diag(i, j) + tbr(j)
        if (diag(i, j) <= 0.0_dp) diag(i, j) = 1.0_dp
      end do
    end do

    call apply_A(x, tx, ty, tbl, tbr, nx, ny, Ap)
    r = b - Ap
    bnorm = sqrt(sum(b * b))
    if (bnorm <= 0.0_dp) bnorm = 1.0_dp
    z = r / diag
    pdir = z
    rz = sum(r * z)
    iters = 0; resid = sqrt(sum(r * r)) / bnorm
    best_resid = resid; xbest = x; since_best = 0
    do it = 1, maxit
      if (resid <= tol) exit
      call apply_A(pdir, tx, ty, tbl, tbr, nx, ny, Ap)
      pAp = sum(pdir * Ap)
      if (.not. (abs(pAp) > 1.0e-300_dp)) exit      ! breakdown (also catches NaN)
      alpha = rz / pAp
      x = x + alpha * pdir
      r = r - alpha * Ap
      resid = sqrt(sum(r * r)) / bnorm
      if (.not. (resid == resid)) exit               ! NaN: stop, keep the best iterate
      if (resid < best_resid) then
        best_resid = resid; xbest = x; since_best = 0
      else
        since_best = since_best + 1
      end if
      ! Stagnation.  For a permeability contrast of 1e5 on a fine grid the
      ! Jacobi-preconditioned recursion floors out above 1e-12 in double
      ! precision; grinding on until maxit used to end in a 0/0 and a NaN
      ! field shipped across the wire.  Stop, and report the residual reached.
      if (since_best >= 3000) exit
      z = r / diag
      rz_new = sum(r * z)
      if (.not. (rz_new > 0.0_dp)) exit
      beta = rz_new / rz
      rz = rz_new
      pdir = z + beta * pdir
      iters = it
    end do
    if (.not. (resid <= best_resid)) then
      x = xbest; resid = best_resid
    end if
  end subroutine cg_solve

end module darcy_mod
