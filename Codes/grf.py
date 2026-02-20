# -*- coding: utf-8 -*- 
# """
# Jingye Tan
# Created on 2023-10-03
# 

import fenics as fe
import numpy as np
import math
import matplotlib.pyplot as plt
# fe.parameters["form_compiler"]["representation"] = "quadrature"
# fe.parameters["form_compiler"]["quadrature_degree"] = 5
# fe.set_log_level(fe.LogLevel.ERROR)

# mesh = fe.UnitSquareMesh(10, 10)
# V = fe.FunctionSpace(mesh, "CG", 1)

# # Define Gaussian random field parameters
# sigma2 = 0.1
# correlation_length = 0.25
# ##############################


# ndim = 2 
# nu = 2.0 - 0.5 * ndim 
# kappa = np.sqrt(8 * nu) / correlation_length
# scaling_factor = (
#     np.sqrt(sigma2)
#     * np.power(kappa, nu)
#     * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu))
# )
# gamma = float(1.0 / scaling_factor)
# delta = float(np.power(kappa, 2) / scaling_factor)


# trial = fe.TrialFunction(V)
# test = fe.TestFunction(V)

# # Define bilinear forms
# stiffness_form = (
#     fe.Constant(gamma) * fe.inner(fe.grad(trial), fe.grad(test)) * fe.dx(metadata={"quadrature_degree": -1})
# )
# mass_form = fe.Constant(delta) * fe.inner(trial, test) * fe.dx

# robin_coefficient = gamma * fe.Constant(np.sqrt(delta / gamma)) / fe.Constant(1.42)
# robin_form = robin_coefficient * fe.inner(trial, test) * fe.ds

# # Assemble system matrices
# M = fe.assemble(mass_form)
# A = fe.assemble(stiffness_form + mass_form + robin_form)
# # Configure solver
# solver = fe.PETScKrylovSolver("cg", "petsc_amg")
# solver.set_operator(A)
# # Configure quadrature space for sampling
# fe.parameters["form_compiler"]["quadrature_degree"] = -1
# metadata = {"quadrature_degree": 2 * V._ufl_element.degree()}
# fe.parameters["form_compiler"]["representation"] = "quadrature"
# Qh = fe.FunctionSpace(
#     mesh,
#     fe.FiniteElement("Quadrature", V.mesh().ufl_cell(), 2 * V._ufl_element.degree(), quad_scheme="default"),
# )
# ph = fe.TrialFunction(Qh)
# qh = fe.TestFunction(Qh)
# # Assemble quadrature mass matrix
# Mqh = fe.assemble(fe.inner(ph, qh) * fe.dx(metadata=metadata))
# ones = fe.interpolate(fe.Constant(1.0), Qh).vector()
# dMqh = Mqh * ones
# Mqh.zero()
# dMqh.set_local(ones.get_local() / np.sqrt(dMqh.get_local()))
# Mqh.set_diagonal(dMqh)
# # Assemble mixed matrix
# MixedM = fe.assemble(fe.inner(ph, test) * fe.dx(metadata=metadata))
# Amat = fe.as_backend_type(MixedM).mat()
# Bmat = fe.as_backend_type(Mqh).mat()
# foo = Amat.matMult(Bmat)
# # Set local-global maps for the resulting matrix
# rmap, _ = Amat.getLGMap()
# _, cmap = Bmat.getLGMap()
# foo.setLGMap(rmap, cmap)
# # Create square root of the mass matrix
# sqrtM = fe.Matrix(fe.PETScMatrix(foo))
# fe.parameters["form_compiler"]["quadrature_degree"] = 5


# # get 10 samples and plot them
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# axes = axes.flatten()

# for nsample in range(10):
#     # Generate white noise
#     noise = fe.Vector()
#     sqrtM.init_vector(noise, 1)
#     np_noise = np.random.normal(0, 1, noise.local_size())
#     noise.set_local(np_noise)

#     # Solve for the sample
#     sample = fe.Function(V, name="sample")
#     rhs = sqrtM * noise
#     solver.solve(sample.vector(), rhs)

#     plt.sca(axes[nsample])
#     fe.plot(sample)
#     axes[nsample].set_title(f"Sample {nsample + 1}")
#     axes[nsample].axis("off")


# plt.tight_layout()
# plt.show()




# def generate_grf_samples(
#     mesh, 
#     V, 
#     sigma2=0.1, 
#     correlation_length=0.25, 
#     num_samples=1, 
#     degree=1,
#     plot=False
# ):
#     """
#     Generate Gaussian random field samples using the spectral method on a FEniCS mesh.

#     Parameters:
#     -----------
#     mesh : fe.Mesh
#         The FEniCS mesh on which the GRF is defined.
#     V : fe.FunctionSpace
#         The function space to define the GRF.
#     sigma2 : float
#         Variance of the Gaussian random field.
#     correlation_length : float
#         Correlation length of the field.
#     num_samples : int
#         Number of GRF samples to generate.
#     degree : int
#         Degree of the quadrature elements used.
#     plot : bool
#         Whether to plot the generated samples.

#     Returns:
#     --------
#     samples : list of fe.Function
#         List of GRF samples as FEniCS Functions.
#     """

#     ndim = mesh.geometry().dim()
#     nu = 2.0 - 0.5 * ndim
#     kappa = np.sqrt(8 * nu) / correlation_length

#     scaling_factor = (
#         np.sqrt(sigma2)
#         * np.power(kappa, nu)
#         * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu))
#     )
#     gamma = float(1.0 / scaling_factor)
#     delta = float(np.power(kappa, 2) / scaling_factor)

#     fe.parameters["form_compiler"]["representation"] = "quadrature"
#     fe.parameters["form_compiler"]["quadrature_degree"] = 5

#     trial = fe.TrialFunction(V)
#     test = fe.TestFunction(V)

#     stiffness_form = gamma * fe.inner(fe.grad(trial), fe.grad(test)) * fe.dx
#     mass_form = delta * fe.inner(trial, test) * fe.dx
#     robin_coefficient = gamma * np.sqrt(delta / gamma) / 1.42
#     robin_form = robin_coefficient * fe.inner(trial, test) * fe.ds

#     A = fe.assemble(stiffness_form + mass_form + robin_form)
#     solver = fe.PETScKrylovSolver("cg", "petsc_amg")
#     solver.set_operator(A)

#     Qh = fe.FunctionSpace(
#         mesh,
#         fe.FiniteElement("Quadrature", mesh.ufl_cell(), 2 * degree, quad_scheme="default")
#     )
#     ph = fe.TrialFunction(Qh)
#     qh = fe.TestFunction(Qh)
#     Mqh = fe.assemble(fe.inner(ph, qh) * fe.dx(metadata={"quadrature_degree": 2 * degree}))

#     ones = fe.interpolate(fe.Constant(1.0), Qh).vector()
#     dMqh = Mqh * ones
#     Mqh.zero()
#     dMqh.set_local(ones.get_local() / np.sqrt(dMqh.get_local()))
#     Mqh.set_diagonal(dMqh)

#     MixedM = fe.assemble(fe.inner(ph, test) * fe.dx(metadata={"quadrature_degree": 2 * degree}))
#     Amat = fe.as_backend_type(MixedM).mat()
#     Bmat = fe.as_backend_type(Mqh).mat()
#     foo = Amat.matMult(Bmat)

#     rmap, _ = Amat.getLGMap()
#     _, cmap = Bmat.getLGMap()
#     foo.setLGMap(rmap, cmap)

#     sqrtM = fe.Matrix(fe.PETScMatrix(foo))

#     # Sample generation
#     samples = []
#     for _ in range(num_samples):
#         noise = fe.Vector()
#         sqrtM.init_vector(noise, 1)
#         np_noise = np.random.normal(0, 1, noise.local_size())
#         noise.set_local(np_noise)

#         sample = fe.Function(V)
#         rhs = sqrtM * noise
#         solver.solve(sample.vector(), rhs)
#         samples.append(sample)

#     # # Optional plot
#     # if plot:
#     #     import matplotlib.pyplot as plt
#     #     cols = min(5, num_samples)
#     #     rows = (num_samples + cols - 1) // cols
#     #     fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
#     #     axes = axes.flatten() if num_samples > 1 else [axes]
#     #     for i, s in enumerate(samples):
#     #         fe.plot(s, ax=axes[i])
#     #         axes[i].set_title(f"Sample {i+1}")
#     #         axes[i].axis("off")
#     #     plt.tight_layout()
#     #     plt.show()

#     return samples






def generate_grf_samples(mesh, V, sigma2=0.1, correlation_length=0.25,
                         num_samples=1, degree=1, plot=False):

    fc = fe.parameters["form_compiler"]

    # ---- SAVE caller settings (legacy-safe) ----
    try:
        old_repr = fc["representation"]
    except RuntimeError:
        old_repr = None
    try:
        old_qdeg = fc["quadrature_degree"]
    except RuntimeError:
        old_qdeg = None
    try:
        old_opt = fc["optimize"]
    except RuntimeError:
        old_opt = None
    try:
        old_cpp = fc["cpp_optimize"]
    except RuntimeError:
        old_cpp = None

    try:
        # ---- GRF local settings ----
        fc["representation"] = "quadrature"
        fc["quadrature_degree"] = 5
        fc["optimize"] = True
        fc["cpp_optimize"] = True

        ndim = mesh.geometry().dim()
        nu = 2.0 - 0.5 * ndim
        kappa = np.sqrt(8 * nu) / correlation_length

        scaling_factor = (
            np.sqrt(sigma2)
            * np.power(kappa, nu)
            * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu))
        )
        gamma = float(1.0 / scaling_factor)
        delta = float(np.power(kappa, 2) / scaling_factor)

        fe.parameters["form_compiler"]["representation"] = "quadrature"
        fe.parameters["form_compiler"]["quadrature_degree"] = 5

        trial = fe.TrialFunction(V)
        test = fe.TestFunction(V)

        stiffness_form = gamma * fe.inner(fe.grad(trial), fe.grad(test)) * fe.dx
        mass_form = delta * fe.inner(trial, test) * fe.dx
        robin_coefficient = gamma * np.sqrt(delta / gamma) / 1.42
        robin_form = robin_coefficient * fe.inner(trial, test) * fe.ds

        A = fe.assemble(stiffness_form + mass_form + robin_form)
        solver = fe.PETScKrylovSolver("cg", "petsc_amg")
        solver.set_operator(A)

        Qh = fe.FunctionSpace(
            mesh,
            fe.FiniteElement("Quadrature", mesh.ufl_cell(), 2 * degree, quad_scheme="default")
        )
        ph = fe.TrialFunction(Qh)
        qh = fe.TestFunction(Qh)
        Mqh = fe.assemble(fe.inner(ph, qh) * fe.dx(metadata={"quadrature_degree": 2 * degree}))

        ones = fe.interpolate(fe.Constant(1.0), Qh).vector()
        dMqh = Mqh * ones
        Mqh.zero()
        dMqh.set_local(ones.get_local() / np.sqrt(dMqh.get_local()))
        Mqh.set_diagonal(dMqh)

        MixedM = fe.assemble(fe.inner(ph, test) * fe.dx(metadata={"quadrature_degree": 2 * degree}))
        Amat = fe.as_backend_type(MixedM).mat()
        Bmat = fe.as_backend_type(Mqh).mat()
        foo = Amat.matMult(Bmat)

        rmap, _ = Amat.getLGMap()
        _, cmap = Bmat.getLGMap()
        foo.setLGMap(rmap, cmap)

        sqrtM = fe.Matrix(fe.PETScMatrix(foo))

        # Sample generation
        samples = []
        for _ in range(num_samples):
            noise = fe.Vector()
            sqrtM.init_vector(noise, 1)
            np_noise = np.random.normal(0, 1, noise.local_size())
            noise.set_local(np_noise)

            sample = fe.Function(V)
            rhs = sqrtM * noise
            solver.solve(sample.vector(), rhs)
            samples.append(sample)

        return samples

    finally:
        # ---- RESTORE caller settings ----
        if old_repr is not None:
            fc["representation"] = old_repr
        if old_qdeg is not None:
            fc["quadrature_degree"] = old_qdeg
        if old_opt is not None:
            fc["optimize"] = old_opt
        if old_cpp is not None:
            fc["cpp_optimize"] = old_cpp
