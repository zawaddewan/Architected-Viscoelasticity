# -----------------------------------------------------------------------------------
# COUPLED TO STAGGERED APPROACH FOR NON-CONVEX CONSTITUTIVE MODEL [METASTABLE]
# Rollers on the sides and bottom, Disp Controlled Compression Test
# (Modeled in 2D)
# Involves Gaussian Random Field for Bulk Modulus
# Simulation involves entire domain and artificial viscosity
# -----------------------------------------------------------------------------------

from fenics import *
from mshr import *
from grf import generate_grf_samples

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import os
import time

start_time = time.time()

print("\033[1;32m--- Starting the Code ---\033[1;m", flush=True)

# Form Compiler Options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# Solver Parameters
# -----------------------------------------------------------------------------------
# COUPLED SOLVER PARAMETERS
coup_solver_param = {"nonlinear_solver": "snes",
                      "snes_solver": {"maximum_iterations": 100,
                                      "report": True,
                                      "line_search": "bt",
                                      "preconditioner": "hypre_amg",
                                      "linear_solver": "mumps",
                                      "method": "newtonls",
                                      "relative_tolerance": 1e-9,
                                      "absolute_tolerance": 1e-9,
                                      "error_on_nonconvergence": True}}

# STAGGERED SOLVER PARAMETERS
u_solver_param = {"nonlinear_solver": "snes",
                      "snes_solver": {"maximum_iterations": 1000,
                                      "report": True,
                                      "line_search": "bt",
                                      "preconditioner": "hypre_amg",
                                      "linear_solver": "mumps",
                                      "method": "newtonls",
                                      "relative_tolerance": 1e-9,
                                      "absolute_tolerance": 1e-9,
                                      "error_on_nonconvergence": True}}

phi_solver_param = {"nonlinear_solver": "snes",
                      "snes_solver": {"maximum_iterations": 100,
                                      "report": True,
                                      "line_search": "bt",
                                      "preconditioner": "hypre_amg",
                                      "linear_solver": "mumps",
                                      "method": "newtonls",
                                      "relative_tolerance": 1e-9,
                                      "absolute_tolerance": 1e-9,
                                      "error_on_nonconvergence": True}}

# Model Parameters
# -----------------------------------------------------------------------------------
# Set the user parameters, can be parsed from the command line
parameters.parse()
userpar = Parameters("user")
userpar.add("Mu", 1.0)                      # Shear Modulus
userpar.add("Kappa", 1.0)                    # Bulk Modulus
userpar.add("eta", 5.0)                     # Artificial Viscosity
userpar.add("dt", 1.0)                      # Quasi-Timestep
userpar.add("CV", 1e-1)                     # Central Variance of the GRF
userpar.add("corr_ratio", 4.0)              # Corr length to non-local length ratio
userpar.add("Beta", 0.5)
userpar.add("Alpha", 300.0)
userpar.parse()

# Set the material properties
Mu = userpar["Mu"]
Kappa = userpar["Kappa"]
eta = userpar["eta"]
dt = userpar["dt"]
CV = userpar["CV"]
corr_ratio = userpar["corr_ratio"]
Beta = userpar["Beta"]
Alpha = userpar["Alpha"]
tol = 1e-12

# Name of the file
name = "Metastable"
simparam1 = "_mu_%.1f" % (Mu)
simparam2 = "_kappa_%.1f" % (Kappa)
simparam3 = "_eta_%.1f" % (eta)
# simparam4 = "_CV_%.3e" % (CV)

# # For Unstructured Mesh
# savedir = f"../Results/GRF_CV_{CV}/Unstruct_Mesh/Corr_Length_=_{corr_ratio}*Avg_Length/Realization_IV"

# For Structured Mesh
savedir = f"../Results/GRF_CV_{CV}/Struct_Mesh/LnU_Corr_Length_=_{corr_ratio}*Avg_Length/Realization_I"

# Define the Mesh
# -----------------------------------------------------------------------------------
# # Unstructured Mesh
# domain = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))

# # Generate the Mesh
# resolution = 64
# mesh = generate_mesh(domain, resolution)

# Structured Mesh
nx, ny = 64, 64
p0, p1 = Point(0.0, 0.0), Point(1.0, 1.0)

mesh = RectangleMesh(p0, p1, nx, ny)

# # Visualize the Mesh
# plot(mesh)
# plt.show()
# exit()

# Mesh Statistics
h = [Cell(mesh, cell.index()).circumradius() for cell in cells(mesh)]

# Get statistics
max_length = max(h)
min_length = min(h)
avg_length = sum(h) / len(h)

print(f"\033[1;36m Max element length: {max_length}\033[1;m")
print(f"\033[1;36m Min element length: {min_length}\033[1;m")
print(f"\033[1;36m Avg element length: {avg_length}\033[1;m")

# Define the subdomains, boundaries and points
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)

# Measure/Redefinition of dx and ds according to subdomains and boundaries
dx = Measure("dx")(subdomain_data = subdomains)
ds = Measure("ds")(subdomain_data = boundaries)

# Define the Boundaries
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)

Top = Top()
Bottom = Bottom()
Left = Left()
Right = Right()

# Mark the boundaries
boundaries.set_all(0)
Top.mark(boundaries, 1)
Bottom.mark(boundaries, 2)
Left.mark(boundaries, 3)
Right.mark(boundaries, 4)

# # Visualization for asserting the boundaries
# file = XDMFFile("boundaries.xdmf")
# file.write(boundaries)
# exit()

# Define the Function Spaces
# -----------------------------------------------------------------------------------
# Function Spaces for Jacobian, Strain Energy, Stress and Strain
JJ = FunctionSpace(mesh, 'DG', 0)                   # For Jacobian
EE = FunctionSpace(mesh, 'DG', 0)                   # For Strain Energy
TT = TensorFunctionSpace(mesh, 'DG', 0)             # For Stresses
ST = TensorFunctionSpace(mesh, 'DG', 0)             # For Strains

# Function Space for unknowns - displacement, u and surrogate J, \Tilde{J} called phi here
# Define the Taylor-Hood elements
# Second Order Quadratic Interpolation for displacement
P2 = VectorFunctionSpace(mesh, "Lagrange", 2)

# First Order Linear Interpolation for surrogate J, phi
P1 = FunctionSpace(mesh, "Lagrange", 1)

# Return the UFL elements of function spaces, P1 and P2
P2elem = P2.ufl_element()
P1elem = P1.ufl_element()

# Define the Mixed Function Space
V = FunctionSpace(mesh, P2elem*P1elem)

# Define a function space for Bulk Modulus, kappa
K = FunctionSpace(mesh, 'CG', 1)
M = FunctionSpace(mesh, 'CG', 1)
A = FunctionSpace(mesh, 'CG', 1)
B = FunctionSpace(mesh, 'CG', 1)

# Define the functions in the Mixed Function Space, V
# -----------------------------------------------------------------------------------
dw = TrialFunction(V)                   # Incremental Trial Function
v = TestFunction(V)                     # Test Function
w = Function(V)                         # Solution for displacement, u and surrogate J, phi
w0 = Function(V)                        # Solution for previous step

# Split the test function and unknowns (produces a shallow, not a deep copy)
(v_u, v_phi) = split(v)
(u, phi) = split(w)
(u0, phi0) = split(w0)

# Boundary Conditions
# -----------------------------------------------------------------------------------
# Rollers on the sides and bottom
bcl = DirichletBC(V.sub(0).sub(0), Constant(0.0), Left)
bcr = DirichletBC(V.sub(0).sub(0), Constant(0.0), Right)
bcb = DirichletBC(V.sub(0).sub(1), Constant(0.0), Bottom)
ucomp = Expression(("-inc"), inc = 0.0, degree = 0)
bct = DirichletBC(V.sub(0).sub(1), ucomp, Top)

bc = [bcl, bcr, bcb, bct]

# Initial Conditions (IC)
# -----------------------------------------------------------------------------------
class InitialCondition(UserExpression):
    def eval(self, value, x):
        # Initial zero displacement
        value[0] = 0.0
        value[1] = 0.0

        # Surrogate J, phi being 1.0 initially
        value[2] = 1.0

    def value_shape(self):
        return (3,)
    
init = InitialCondition(degree=1)
w0.interpolate(init)
w.interpolate(init)

# Setting the GRF for Kappa
# -----------------------------------------------------------------------------------
kappa_bar = float(Kappa)     # desired mean bulk modulus (your userpar)
CV_target = CV               # e.g., 20% contrast; lower this to calm variation

# Convert desired CV to variance of the Gaussian seed Z so that
# kappa = kappa_bar * exp(Z - 0.5*Var(Z)) has mean=kappa_bar and CV = sqrt(exp(Var(Z))-1)
sigma2_Z = float(np.log(1.0 + CV_target**2))

# If you want correlation length relative to domain size, compute it here:
# (You were using avg_length; keep if that’s your intended ℓ. Otherwise, use domain-span.)
x_min = mesh.coordinates()[:,0].min(); x_max = mesh.coordinates()[:,0].max()
y_min = mesh.coordinates()[:,1].min(); y_max = mesh.coordinates()[:,1].max()
Lx = x_max - x_min; Ly = y_max - y_min
ell = corr_ratio*avg_length            # or e.g., ell = 0.15 * max(Lx, Ly)

# Sample Z ~ GRF(0, sigma2_Z, ℓ) in space Q
samples = generate_grf_samples(mesh, K,
                               sigma2=sigma2_Z,
                               correlation_length=ell,
                               num_samples=1,
                               plot=False)
Z = samples[0]
z = Z.vector().get_local()

# Optional: very gentle winsorization only if you see extreme spikes; otherwise skip it
# z = np.clip(z, -3.5*np.sqrt(sigma2_Z), 3.5*np.sqrt(sigma2_Z))

# Lognormal map with exact mean control (no post scaling step)
kappa = Function(K)
kappa.vector().set_local(kappa_bar * np.exp(z - 0.5*sigma2_Z))
kappa.vector().apply("insert")

# Serial stats
domain_volume = assemble(Constant(1.0) * dx(domain=mesh))
kappa_min = kappa.vector().min()
kappa_max = kappa.vector().max()
global_mean = assemble(kappa * dx) / domain_volume

print(f"\033[1;36m[GRF κ] min={kappa_min:.4f}, max={kappa_max:.4f}, mean={global_mean:.4f} "
      f"(target CV≈{CV_target:.2f})\033[1;m", flush=True)

# Setting the GRF for Mu
# -----------------------------------------------------------------------------------
mu_bar = float(Mu)     # desired mean shear modulus (your userpar)
CV_target = CV            # e.g., 20% contrast; lower this to calm variation

# Convert desired CV to variance of the Gaussian seed Z so that
# kappa = kappa_bar * exp(Z - 0.5*Var(Z)) has mean=kappa_bar and CV = sqrt(exp(Var(Z))-1)
sigma2_Z = float(np.log(1.0 + CV_target**2))

# If you want correlation length relative to domain size, compute it here:
# (You were using avg_length; keep if that’s your intended ℓ. Otherwise, use domain-span.)
x_min = mesh.coordinates()[:,0].min(); x_max = mesh.coordinates()[:,0].max()
y_min = mesh.coordinates()[:,1].min(); y_max = mesh.coordinates()[:,1].max()
Lx = x_max - x_min; Ly = y_max - y_min
ell = corr_ratio*avg_length            # or e.g., ell = 0.15 * max(Lx, Ly)

# Sample Z ~ GRF(0, sigma2_Z, ℓ) in space Q
samples = generate_grf_samples(mesh, M,
                               sigma2=sigma2_Z,
                               correlation_length=ell,
                               num_samples=1,
                               plot=False)
Z = samples[0]
z = Z.vector().get_local()

# Optional: very gentle winsorization only if you see extreme spikes; otherwise skip it
# z = np.clip(z, -3.5*np.sqrt(sigma2_Z), 3.5*np.sqrt(sigma2_Z))

# Lognormal map with exact mean control (no post scaling step)
mu = Function(M)
mu.vector().set_local(mu_bar * np.exp(z - 0.5*sigma2_Z))
mu.vector().apply("insert")

# Serial stats
domain_volume = assemble(Constant(1.0) * dx(domain=mesh))
mu_min = mu.vector().min()
mu_max = mu.vector().max()
global_mean = assemble(mu * dx) / domain_volume

print(f"\033[1;36m[GRF mu] min={mu_min:.4f}, max={mu_max:.4f}, mean={global_mean:.4f} "
      f"(target CV≈{CV_target:.2f})\033[1;m", flush=True)

# Setting the GRF for Alpha
# -----------------------------------------------------------------------------------
Alpha_bar = float(Alpha)     # desired mean shear modulus (your userpar)
CV_target = CV            # e.g., 20% contrast; lower this to calm variation

# Convert desired CV to variance of the Gaussian seed Z so that
# kappa = kappa_bar * exp(Z - 0.5*Var(Z)) has mean=kappa_bar and CV = sqrt(exp(Var(Z))-1)
sigma2_Z = float(np.log(1.0 + CV_target**2))

# If you want correlation length relative to domain size, compute it here:
# (You were using avg_length; keep if that’s your intended ℓ. Otherwise, use domain-span.)
x_min = mesh.coordinates()[:,0].min(); x_max = mesh.coordinates()[:,0].max()
y_min = mesh.coordinates()[:,1].min(); y_max = mesh.coordinates()[:,1].max()
Lx = x_max - x_min; Ly = y_max - y_min
ell = corr_ratio*avg_length            # or e.g., ell = 0.15 * max(Lx, Ly)

# Sample Z ~ GRF(0, sigma2_Z, ℓ) in space Q
samples = generate_grf_samples(mesh, A,
                               sigma2=sigma2_Z,
                               correlation_length=ell,
                               num_samples=1,
                               plot=False)
Z = samples[0]
z = Z.vector().get_local()

# Optional: very gentle winsorization only if you see extreme spikes; otherwise skip it
# z = np.clip(z, -3.5*np.sqrt(sigma2_Z), 3.5*np.sqrt(sigma2_Z))

# Lognormal map with exact mean control (no post scaling step)
alpha0 = Function(A)
alpha0.vector().set_local(Alpha_bar * np.exp(z - 0.5*sigma2_Z))
alpha0.vector().apply("insert")

# Serial stats
domain_volume = assemble(Constant(1.0) * dx(domain=mesh))
alpha_min = alpha0.vector().min()
alpha_max = alpha0.vector().max()
global_mean = assemble(alpha0 * dx) / domain_volume

print(f"\033[1;36m[GRF alpha] min={alpha_min:.4f}, max={alpha_max:.4f}, mean={global_mean:.4f} "
      f"(target CV≈{CV_target:.2f})\033[1;m", flush=True)

# Setting the GRF for Beta
# -----------------------------------------------------------------------------------
Beta_bar = float(Beta)     # desired mean shear modulus (your userpar)
CV_target = CV            # e.g., 20% contrast; lower this to calm variation

# Convert desired CV to variance of the Gaussian seed Z so that
# kappa = kappa_bar * exp(Z - 0.5*Var(Z)) has mean=kappa_bar and CV = sqrt(exp(Var(Z))-1)
sigma2_Z = float(np.log(1.0 + CV_target**2))

# If you want correlation length relative to domain size, compute it here:
# (You were using avg_length; keep if that’s your intended ℓ. Otherwise, use domain-span.)
x_min = mesh.coordinates()[:,0].min(); x_max = mesh.coordinates()[:,0].max()
y_min = mesh.coordinates()[:,1].min(); y_max = mesh.coordinates()[:,1].max()
Lx = x_max - x_min; Ly = y_max - y_min
ell = corr_ratio*avg_length            # or e.g., ell = 0.15 * max(Lx, Ly)

# Sample Z ~ GRF(0, sigma2_Z, ℓ) in space Q
samples = generate_grf_samples(mesh, B,
                               sigma2=sigma2_Z,
                               correlation_length=ell,
                               num_samples=1,
                               plot=False)
Z = samples[0]
z = Z.vector().get_local()

# Optional: very gentle winsorization only if you see extreme spikes; otherwise skip it
# z = np.clip(z, -3.5*np.sqrt(sigma2_Z), 3.5*np.sqrt(sigma2_Z))

# Lognormal map with exact mean control (no post scaling step)
beta = Function(B)
beta.vector().set_local(Beta_bar * np.exp(z - 0.5*sigma2_Z))
beta.vector().apply("insert")

# Serial stats
domain_volume = assemble(Constant(1.0) * dx(domain=mesh))
beta_min = beta.vector().min()
beta_max = beta.vector().max()
global_mean = assemble(beta * dx) / domain_volume

print(f"\033[1;36m[GRF beta] min={beta_min:.4f}, max={beta_max:.4f}, mean={global_mean:.4f} "
      f"(target CV≈{CV_target:.2f})\033[1;m", flush=True)

# # Plot the field asserting the Bulk Modulus over the domain
# # -----------------------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot the field
# c = plot(mu, cmap='viridis')  # Don't pass axes=ax to avoid contour warning
# ax = plt.gca()

# # Set axis labels
# ax.set_xlabel("x", fontsize=25)
# ax.set_ylabel("y", fontsize=25)

# # Create vertical colorbar
# cbar = fig.colorbar(c, ax=ax, orientation='vertical')
# cbar.set_label("")  # Clear default label

# # Optional: remove default tick label position indicators if needed
# cbar.ax.xaxis.set_ticks_position('none')

# # Manually add horizontal label at same level as x-axis
# # You can tweak the x and y coordinates if needed
# fig.text(0.85, 0.07, r"$\mu(y)$", ha='center', va='center', fontsize=25)

# plt.tight_layout()
# plt.show()
# exit()

# Kinematics
# -----------------------------------------------------------------------------------
d = len(u)                          # Spatial Dimension
I = Identity(d)                     # Identity Tensor
F = I + grad(u)                     # Deformation Gradient
C = F.T*F                           # Right Cauchy-Green Tensor

# Invariants
Ic = tr(C) + 1.0                    # First Invariant
J = det(F)                          # Third Invariant

# Define the Strain Energy Density Function
# -----------------------------------------------------------------------------------
# Original Parameters
mu0 = Constant(2.0)
kappa0 = Constant(2.0)
# alpha0 = Constant(300.0)
# beta = Constant(0.5)
c0 = Constant(230.0)
l20 = Constant((avg_length)**2)

# Ratios
r_k = kappa0 / mu0
r_a = alpha0 / mu0
r_c = c0 / mu0
r_l2 = l20 / mu0

# Convex part of the strain energy density
psiC = (mu/2.0)*(Ic - 3.0 - 2.0*ln(J)) + kappa*(r_k/2.0)*(ln(J))**2

# Non-Convex part
psiNC = (r_a/2.0)*((1/2)*(1 - phi)**2 - beta*(1 - phi))**2
psi_coup = r_c*((J - phi)**2) + r_l2*dot(grad(phi), grad(phi)) 

# Total Helmholtz Free Energy Density Function
psi = psiC + psiNC + psi_coup

# Define the Gibbs Free Energy (similar to total Potential Energy), Nominal Stress and Lagrange-Green Strain
# -----------------------------------------------------------------------------------
Pi = psi*dx

# Define the Nominal Stress (first Piola-Kirchoff Stress)
def P(u, phi):
    return mu*(F - inv(F.T)) + r_k*inv(F.T)*ln(J) + 2*r_c*(J - phi)*J*inv(F.T)

# Define the Strain (Lagrange-Green Strain)
def E(u):
    return (1/2)*(C - I)

# Define the Weak Form
# -----------------------------------------------------------------------------------
WF_coup = derivative(Pi, w, v)
WF_coup += eta * ((phi - phi0)/dt) * v_phi * dx                      # Incorporating Artificial Viscosity

Jacobian_coup = derivative(WF_coup, w, dw)

# Define the Problem
coup_problem = NonlinearVariationalProblem(WF_coup, w, bc, J=Jacobian_coup)
coup_solver = NonlinearVariationalSolver(coup_problem)
coup_solver.parameters.update(coup_solver_param)

# Save the Results
results = XDMFFile(savedir + "/" + name + simparam1 + simparam2 + simparam3 + ".xdmf")
results.parameters["flush_output"] = True
results.parameters["functions_share_mesh"] = True

# Set up the STAGGERED APPROACH
# -----------------------------------------------------------------------------------
# Define the functions for the unknowns - displacement, u and surrogate J, phi
du = TrialFunction(P2)
test_u = TestFunction(P2)
ustag = Function(P2)
ustag_previous = Function(P2)

dphi = TrialFunction(P1)
test_phi = TestFunction(P1)
phistag = Function(P1)
phistag_dummy = Function(P1)
phistag_previous = Function(P1)

# Boundary Conditions
# -----------------------------------------------------------------------------------
bcl_stag = DirichletBC(P2.sub(0), Constant(0.0), Left)
bcb_stag = DirichletBC(P2.sub(1), Constant(0.0), Bottom)
bcr_stag = DirichletBC(P2.sub(0), Constant(0.0), Right)
bct_stag = DirichletBC(P2.sub(1), ucomp, Top)

bc_stag = [bcl_stag, bcb_stag, bcr_stag, bct_stag]

# Kinematics
# -----------------------------------------------------------------------------------
dstag = len(ustag)
Istag = Identity(dstag)
Fstag = Istag + grad(ustag)
Cstag = Fstag.T*Fstag

# Invariants
Icstag = tr(Cstag) + 1.0
Jstag = det(Fstag)

# Define the Strain Energy Density Function
# -----------------------------------------------------------------------------------
# Convex part of the strain energy density
psiC_stag = (mu/2.0)*(Icstag - 3.0 - 2.0*ln(Jstag)) + kappa*(r_k/2.0)*(ln(Jstag))**2

# Non-Convex part
psiNC_stag = (r_a/2.0)*((1/2)*(1 - phistag)**2 - beta*(1 - phistag))**2
psi_coup_stag = r_c*((Jstag - phistag)**2) + r_l2*dot(grad(phistag), grad(phistag)) 

psi_stag = psiC_stag + psiNC_stag + psi_coup_stag

Pi_stag = psi_stag * dx

# Define the Variational Problem
# -----------------------------------------------------------------------------------
WF_u = derivative(Pi_stag, ustag, test_u)

Jacobian_u = derivative(WF_u, ustag, du)

WF_phi = derivative(Pi_stag, phistag, test_phi)
WF_phi += eta * ((phistag - phistag_previous)/dt) * test_phi * dx

Jacobian_phi = derivative(WF_phi, phistag, dphi)

# Set up the problems and the solvers for individual unknowns
problem_u = NonlinearVariationalProblem(WF_u, ustag, bc_stag, J=Jacobian_u)
solver_u = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(u_solver_param)

problem_phi = NonlinearVariationalProblem(WF_phi, phistag, J=Jacobian_phi)
solver_phi = NonlinearVariationalSolver(problem_phi)
solver_phi.parameters.update(phi_solver_param)

# Solver Loop
# -----------------------------------------------------------------------------------
# Loading and Unloading
ind_steps = np.concatenate([
    np.linspace(0, 0.8, 500),  # Loading phase
    np.linspace(0.8, 0, 500)   # Unloading phase
])

# Store the values of traction in y-direction
traction_list = []     

max_stag_iters = 500
AM_tolerance = 1e-3

assigner_w = FunctionAssigner(V, [P2, P1])
assigner_w0 = FunctionAssigner(V, [P2, P1])

for i, inc in enumerate(ind_steps):
    
    print(f"\033[1;32m--- Step {i+1}/{len(ind_steps)}: Applying indentation: {inc:.4f} ---\033[1;m", flush=True)
    
    ucomp.inc = inc

    try:
        coup_solver.solve()
        
    except RuntimeError as e:
        (ucoup, phicoup) = w.split(deepcopy=True)
        (ucoup_prev, phicoup_prev) = w0.split(deepcopy=True)

        ustag.assign(interpolate(ucoup, P2))
        ustag_previous.assign(interpolate(ucoup_prev, P2))
        phistag.assign(interpolate(phicoup, P1))
        phistag_previous.assign(interpolate(phicoup_prev, P1))
        
        print(f"\033[1;32m--- Switching to STAGGERED solver at step {i+1}, indentation = {inc:.4f} ---\033[1;m", flush=True)

        # Initialize
        iteration = 1
        err_phi = 1.0

        while err_phi > AM_tolerance and iteration < max_stag_iters:
            print(f"\033[1;32m--- Staggered DISPLACEMENT Solver for indentation: {inc:.4f} ---\033[1;m", flush=True)
            solver_u.solve()
            print(f"\033[1;32m--- Staggered SURROGATE J Solver for indentation: {inc:.4f} ---\033[1;m", flush=True)
            solver_phi.solve()

            # Error Check
            phi_error = phistag.vector() - phistag_dummy.vector()
            err_phi = phi_error.norm('linf')
            
            print("\033[1;32m--- AM Iteration: {0:3d},  phi_error: {1:>14.8f}, indentation: {2:.4f} ---\033[1;m".format(iteration, err_phi, inc), flush=True)

            phistag_dummy.assign(phistag)
            iteration += 1

        # Assign the solution of staggered to coupled spaces
        w_u = interpolate(ustag, P2)
        w_phi = interpolate(phistag, P1)
        w0_u = interpolate(ustag_previous, P2)
        w0_phi = interpolate(phistag_previous, P1)

        # Assign into the mixed functions using FunctionAssigner
        assigner_w.assign(w, [w_u, w_phi])
        assigner_w0.assign(w0, [w0_u, w0_phi])

        # Sanity Check
        u_diff = np.abs(interpolate(w.sub(0), P2).vector().get_local() - ustag.vector().get_local())
        phi_diff = np.abs(interpolate(w.sub(1), P1).vector().get_local() - phistag.vector().get_local())

        if np.max(u_diff) > 1e-14 or np.max(phi_diff) > 1e-14:
            raise RuntimeError("Staggered to Coupled assignment failed: mismatch detected")

    # Update the solution
    w0.vector()[:] = w.vector()[:]
    (u, phi) = w.split()

    # Postprocessing and save
    JScalar = project(J, JJ)
    EScalar = project(psi, EE)
    PTensor = project(P(u, phi), TT)
    STensor = project(E(u), ST)

    P_22 = P(u, phi)[1, 1]
    traction = assemble(P_22*ds(1))
    traction_list.append(-1 * traction)

    beta.rename("Beta", "beta")
    alpha0.rename("Alpha", "alpha0")
    kappa.rename("Bulk Modulus", "kappa")
    mu.rename("Shear Modulus", "mu")
    u.rename("Displacement", "u")
    phi.rename("Densification Parameter", "phi")
    JScalar.rename("Jacobian", "J")
    EScalar.rename("Strain Energy", "psi")
    PTensor.rename("Nominal Stress", "P")
    STensor.rename("Strain", "epsilon")

    results.write(beta, i)
    results.write(alpha0, i)
    results.write(kappa, i)
    results.write(mu, i)
    results.write(u, i)
    results.write(phi, i)
    results.write(JScalar, i)
    results.write(EScalar, i)
    results.write(PTensor, i)
    results.write(STensor, i)

# -----------------------------------------------------------------------------------
# Post-Processing Section
# -----------------------------------------------------------------------------------
# Split the data
mid = len(ind_steps) // 2
disp_loading = ind_steps[:mid]
force_loading = traction_list[:mid]
disp_unloading = ind_steps[mid:]
force_unloading = traction_list[mid:]

# Plotting Loading vs Unloading
plt.figure(figsize=(7, 5))
plt.plot(disp_loading, force_loading, label="Loading", linewidth=2.5)
plt.plot(disp_unloading, force_unloading, label="Unloading", linewidth=2.5, linestyle='--')

plt.xlabel("Displacement (Top)", fontsize=14)
plt.ylabel("Reaction Force", fontsize=14)
plt.title("Traction vs Displacement (Loading-Unloading)", fontsize=16, pad=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)

# Legend inside plot
plt.legend(loc='best', fontsize=12, frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout and save
plt.tight_layout()
plt.savefig(savedir + "/" + "Traction_vs_Displacement.png", dpi=300)

# # Dataset saving
# # For Unstructured Mesh
# csv_file = f"../Dataset/GRF_CV_{CV}/Unstruct_Mesh/Corr_Length_=_{corr_ratio}*Avg_Length/Realization_IV.csv"

# For Structured Mesh
csv_file = f"../Dataset/GRF_CV_{CV}/Struct_Mesh/LnU_Corr_Length_=_{corr_ratio}*Avg_Length/Realization_I.csv"

# Ensure directory exists
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Displacement", "ReactionForce", "Phase"])
    for d, t in zip(disp_loading, force_loading):
        writer.writerow([d, t, "Loading"])
    for d, t in zip(disp_unloading, force_unloading):
        writer.writerow([d, t, "Unloading"])

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds", flush=True)
