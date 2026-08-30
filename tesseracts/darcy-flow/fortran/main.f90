! Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
!
! Driver for the Darcy Tesseract.  Reads a binary request from stdin-named file,
! writes a binary response.  Two modes:
!   mode = 0 : forward only          -> p, ux, uy, flux
!   mode = 1 : forward + VJP         -> p, ux, uy, flux, gamma_bar, mu_bar
!
! Binary layout (little-endian float64 / int32), request:
!   int32   nx, ny, mode, maxit
!   float64 dx, dy, pin, kmin, kmax, qramp, tol
!   float64 gamma(nx*ny), mu(nx*ny)
!   [mode 1 only] float64 pbar(nx*ny), uxbar(nx*ny), uybar(nx*ny), fluxbar
!
! Response:
!   int32   iters                  (FORWARD solve)
!   float64 resid                  (FORWARD solve)
!   float64 p(nx*ny), ux(nx*ny), uy(nx*ny), flux
!   [mode 1 only] int32 adj_iters; float64 adj_resid
!   [mode 1 only] float64 gamma_bar(nx*ny), mu_bar(nx*ny)
!
! The adjoint solve reports its OWN iteration count and residual.  They used
! to overwrite the forward solve's, so a VJP call silently reported the
! adjoint's numbers under the forward's name and nothing anywhere checked
! that the adjoint linear solve had converged at all -- a silent route to a
! wrong gradient, which is the one thing this component must never produce.

program darcy_main
  use darcy_mod
  implicit none

  character(len=1024) :: infile, outfile
  integer :: nx, ny, mode, maxit, iters, adj_iters, i, j, u_in, u_out
  real(dp) :: dx, dy, pin, kmin, kmax, qramp, tol, resid, adj_resid, flux, fluxbar

  real(dp), allocatable :: gamma(:,:), mu(:,:), lam(:,:), kap(:,:), dkap(:,:)
  real(dp), allocatable :: tx(:,:), ty(:,:), tbl(:), tbr(:)
  real(dp), allocatable :: p(:,:), ux(:,:), uy(:,:)
  real(dp), allocatable :: b(:,:), w(:,:), g(:,:)
  real(dp), allocatable :: pbar(:,:), uxbar(:,:), uybar(:,:)
  real(dp), allocatable :: qxe(:,:), qye(:,:), qxebar(:,:), qyebar(:,:)
  real(dp), allocatable :: txbar(:,:), tybar(:,:), tblbar(:), tbrbar(:)
  real(dp), allocatable :: lambar(:,:), gambar(:,:), mubar(:,:)
  real(dp) :: a_, b_, den, dTa, dTb, geomx, geomy, kk, dk

  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)

  open(newunit=u_in, file=trim(infile), form='unformatted', access='stream', status='old')
  read(u_in) nx, ny, mode, maxit
  read(u_in) dx, dy, pin, kmin, kmax, qramp, tol

  allocate(gamma(nx,ny), mu(nx,ny), lam(nx,ny), kap(nx,ny), dkap(nx,ny))
  allocate(tx(nx,ny), ty(nx,ny), tbl(ny), tbr(ny))
  allocate(p(nx,ny), ux(nx,ny), uy(nx,ny), b(nx,ny))
  allocate(qxe(0:nx,ny), qye(nx,0:ny))

  read(u_in) gamma
  read(u_in) mu

  if (mode == 1) then
    allocate(pbar(nx,ny), uxbar(nx,ny), uybar(nx,ny))
    read(u_in) pbar
    read(u_in) uxbar
    read(u_in) uybar
    read(u_in) fluxbar
  end if
  close(u_in)

  ! ---------------- forward ----------------
  do j = 1, ny
    do i = 1, nx
      call kappa_of_gamma(gamma(i,j), kmin, kmax, qramp, kk, dk)
      kap(i,j)  = kk
      dkap(i,j) = dk
      lam(i,j)  = kk / mu(i,j)
    end do
  end do

  call face_trans(lam, nx, ny, dx, dy, tx, ty)
  ! half-cell transmissibility to the Dirichlet boundaries
  do j = 1, ny
    tbl(j) = 2.0_dp * lam(1, j)  * (dy / dx)
    tbr(j) = 2.0_dp * lam(nx, j) * (dy / dx)
  end do

  b = 0.0_dp
  do j = 1, ny
    b(1, j) = tbl(j) * pin
  end do

  adj_iters = 0
  adj_resid = 0.0_dp
  p = 0.0_dp
  call cg_solve(b, tx, ty, tbl, tbr, nx, ny, tol, maxit, p, iters, resid)

  ! face fluxes (extended, including boundary faces)
  qxe = 0.0_dp; qye = 0.0_dp
  do j = 1, ny
    qxe(0, j) = tbl(j) * (pin - p(1, j))
    do i = 1, nx - 1
      qxe(i, j) = tx(i, j) * (p(i, j) - p(i + 1, j))
    end do
    qxe(nx, j) = tbr(j) * p(nx, j)
  end do
  do j = 1, ny - 1
    do i = 1, nx
      qye(i, j) = ty(i, j) * (p(i, j) - p(i, j + 1))
    end do
  end do

  do j = 1, ny
    do i = 1, nx
      ux(i, j) = 0.5_dp * (qxe(i - 1, j) + qxe(i, j)) / dy
      uy(i, j) = 0.5_dp * (qye(i, j - 1) + qye(i, j)) / dx
    end do
  end do

  flux = 0.0_dp
  do j = 1, ny
    flux = flux + qxe(0, j)
  end do

  ! ---------------- adjoint (hand-derived) ----------------
  if (mode == 1) then
    allocate(qxebar(0:nx,ny), qyebar(nx,0:ny))
    allocate(txbar(nx,ny), tybar(nx,ny), tblbar(ny), tbrbar(ny))
    allocate(lambar(nx,ny), gambar(nx,ny), mubar(nx,ny), w(nx,ny), g(nx,ny))
    qxebar = 0.0_dp; qyebar = 0.0_dp
    txbar = 0.0_dp; tybar = 0.0_dp; tblbar = 0.0_dp; tbrbar = 0.0_dp

    ! 1) cotangents on face fluxes, from ux/uy/flux
    do j = 1, ny
      do i = 1, nx
        qxebar(i - 1, j) = qxebar(i - 1, j) + 0.5_dp * uxbar(i, j) / dy
        qxebar(i, j)     = qxebar(i, j)     + 0.5_dp * uxbar(i, j) / dy
        qyebar(i, j - 1) = qyebar(i, j - 1) + 0.5_dp * uybar(i, j) / dx
        qyebar(i, j)     = qyebar(i, j)     + 0.5_dp * uybar(i, j) / dx
      end do
    end do
    do j = 1, ny
      qxebar(0, j) = qxebar(0, j) + fluxbar
    end do

    ! 2) dJ/dp : direct pbar, plus the explicit p-dependence of the fluxes
    g = pbar
    do j = 1, ny
      g(1, j)  = g(1, j)  - tbl(j) * qxebar(0, j)
      g(nx, j) = g(nx, j) + tbr(j) * qxebar(nx, j)
      do i = 1, nx - 1
        g(i, j)     = g(i, j)     + tx(i, j) * qxebar(i, j)
        g(i + 1, j) = g(i + 1, j) - tx(i, j) * qxebar(i, j)
      end do
    end do
    do j = 1, ny - 1
      do i = 1, nx
        g(i, j)     = g(i, j)     + ty(i, j) * qyebar(i, j)
        g(i, j + 1) = g(i, j + 1) - ty(i, j) * qyebar(i, j)
      end do
    end do

    ! 3) adjoint solve  A w = g   (A is symmetric)
    w = 0.0_dp
    call cg_solve(g, tx, ty, tbl, tbr, nx, ny, tol, maxit, w, adj_iters, adj_resid)

    ! 4) cotangents on transmissibilities
    !    (a) explicit, through the flux definitions
    do j = 1, ny
      tblbar(j) = tblbar(j) + qxebar(0, j)  * (pin - p(1, j))
      tbrbar(j) = tbrbar(j) + qxebar(nx, j) * p(nx, j)
      do i = 1, nx - 1
        txbar(i, j) = txbar(i, j) + qxebar(i, j) * (p(i, j) - p(i + 1, j))
      end do
    end do
    do j = 1, ny - 1
      do i = 1, nx
        tybar(i, j) = tybar(i, j) + qyebar(i, j) * (p(i, j) - p(i, j + 1))
      end do
    end do
    !    (b) implicit, -w^T dR/dT
    do j = 1, ny
      tblbar(j) = tblbar(j) - w(1, j)  * (p(1, j) - pin)
      tbrbar(j) = tbrbar(j) - w(nx, j) * p(nx, j)
      do i = 1, nx - 1
        txbar(i, j) = txbar(i, j) - (w(i, j) - w(i + 1, j)) * (p(i, j) - p(i + 1, j))
      end do
    end do
    do j = 1, ny - 1
      do i = 1, nx
        tybar(i, j) = tybar(i, j) - (w(i, j) - w(i, j + 1)) * (p(i, j) - p(i, j + 1))
      end do
    end do

    ! 5) harmonic-mean chain rule: dT_f/dlambda on each side
    lambar = 0.0_dp
    geomx = dy / dx
    geomy = dx / dy
    do j = 1, ny
      do i = 1, nx - 1
        a_ = lam(i, j); b_ = lam(i + 1, j)
        den = max(a_ + b_, 1.0e-300_dp)
        dTa = 2.0_dp * b_ * b_ / (den * den) * geomx
        dTb = 2.0_dp * a_ * a_ / (den * den) * geomx
        lambar(i, j)     = lambar(i, j)     + txbar(i, j) * dTa
        lambar(i + 1, j) = lambar(i + 1, j) + txbar(i, j) * dTb
      end do
    end do
    do j = 1, ny - 1
      do i = 1, nx
        a_ = lam(i, j); b_ = lam(i, j + 1)
        den = max(a_ + b_, 1.0e-300_dp)
        dTa = 2.0_dp * b_ * b_ / (den * den) * geomy
        dTb = 2.0_dp * a_ * a_ / (den * den) * geomy
        lambar(i, j)     = lambar(i, j)     + tybar(i, j) * dTa
        lambar(i, j + 1) = lambar(i, j + 1) + tybar(i, j) * dTb
      end do
    end do
    do j = 1, ny
      lambar(1, j)  = lambar(1, j)  + tblbar(j) * 2.0_dp * geomx
      lambar(nx, j) = lambar(nx, j) + tbrbar(j) * 2.0_dp * geomx
    end do

    ! 6) lambda = kappa(gamma)/mu  ->  gamma_bar, mu_bar
    do j = 1, ny
      do i = 1, nx
        gambar(i, j) = lambar(i, j) * dkap(i, j) / mu(i, j)
        mubar(i, j)  = -lambar(i, j) * kap(i, j) / (mu(i, j) * mu(i, j))
      end do
    end do
  end if

  ! ---------------- write ----------------
  open(newunit=u_out, file=trim(outfile), form='unformatted', access='stream', status='replace')
  write(u_out) iters
  write(u_out) resid
  write(u_out) p
  write(u_out) ux
  write(u_out) uy
  write(u_out) flux
  if (mode == 1) then
    write(u_out) adj_iters
    write(u_out) adj_resid
    write(u_out) gambar
    write(u_out) mubar
  end if
  close(u_out)

end program darcy_main
