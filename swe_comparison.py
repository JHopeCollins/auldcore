import firedrake as fd
from petsc4py import PETSc

# asQ utils
from utils import mg
from utils import units
from utils.planets import earth
from utils.shallow_water import nonlinear as swe
from utils.shallow_water.williamson1992 import case5 as case

PETSc.Sys.popErrorHandler()

import argparse
parser = argparse.ArgumentParser(
    description='Williamson 5 testcase for augmented Lagrangian solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--nt', type=int, default=1, help='Number of timesteps')
parser.add_argument('--save_freq', type=float, default=10, help='Number of timesteps between file outputs.')
parser.add_argument('--gamma', type=float, default=1.0e4, help='Augmented Lagrangian scaling parameter.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--snes_atol', type=float, default=1e0, help='Absolute tolerance for SNES.')
parser.add_argument('--snes_rtol', type=float, default=1e-8, help='Relative tolerance for SNES.')
parser.add_argument('--kspschur', type=int, default=40, help='Max number of KSP iterations on the Schur complement.')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels.')
parser.add_argument('--patch', type=str, default='star', help='Patch type for MG smoother.')
parser.add_argument('--tlblock', type=str, default='mg', help='Solver for the velocity-velocity block. mg==Multigrid with patchPC, lu==direct solver with MUMPS.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

mesh = mg.icosahedral_mesh(earth.radius,
                           base_level=args.base_level,
                           degree=1,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level)
x = fd.SpatialCoordinate(mesh)

degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
W = fd.MixedFunctionSpace((V1, V2))

# case parameters

dt = units.hour*args.dt
dT = fd.Constant(dt)

H = case.H0
g = earth.Gravity

f = case.coriolis_expression(*x)
b = fd.Function(V2, name="Topography").interpolate(case.topography_expression(*x))

# solution vectors

Un = fd.Function(W)
Un1 = fd.Function(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Un1)

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

gamma = fd.Constant(args.gamma)
One = fd.Function(V2).assign(1.0)
half = fd.Constant(0.5)


def ufunc(v, u, h):
    return swe.form_function_velocity(mesh, g, b, f, u, h, v)

def hfunc(phi, u, h):
    return swe.form_function_depth(mesh, u, h, phi)

def umass(v, u):
    return swe.form_mass_u(mesh, u, v)

def hmass(phi, h):
    return swe.form_mass_h(mesh, h, phi)

def ueq(v, u0, u1, h0, h1):
    form = (
    umass(v, u1 - u0)
    + half*dT*ufunc(v, u0, h0)
    + half*dT*ufunc(v, u1, h1)
    )
    return form

def heq(phi, u0, u1, h0, h1):
    form =(
    hmass(phi, h1 - h0)
    + half*dT*hfunc(phi, u0, h0)
    + half*dT*hfunc(phi, u1, h1)
    )
    return form

testeqn = (
    ueq(v, u0, u1, h0, h1)
    + heq(phi, u0, u1, h0, h1)
)

aleqn = (
    gamma*heq(fd.div(v), u0, u1, h0, h1)
)

eqn = testeqn + aleqn

sparameters = {
    "mat_type":"matfree",
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': args.snes_atol,
        'rtol': args.snes_rtol,
    },
    "ksp_type": "fgmres",
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        "gmres_modifiedgramschmidt": None,
    },
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

lu_parameters = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

bottomright_mass = {
    "ksp_type": "gmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_max_it": args.kspschur,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright_mass

patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': args.patch,
        'local_type': 'additive',
        'precompute_element_tensors': True,
        'symmetrise_sweep': False,
        'dense_inverse': True,
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }
}

mg_parameters = {
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': args.kspmg,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': lu_parameters
}

topleft_MG = {
    'ksp_type': 'fgmres',
    'ksp_max_it': 1,
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

if args.tlblock == "mg":
    sparameters["fieldsplit_0"] = topleft_MG
elif args.tlblock=="lu":
    sparameters["fieldsplit_0"] = lu_parameters
else:
    raise ValueError("Unrecognised tlblock argument")

ref_sparams = {
    "mat_type":"matfree",
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': args.snes_atol,
        'rtol': args.snes_rtol,
    },
    "ksp_type": "fgmres",
    'ksp': {
        'monitor': None,
        'converged_reason': None,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

nprob = fd.NonlinearVariationalProblem(eqn, Un1)
ctx = {"mu": gamma*2/(g*dt)}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters,
                                        appctx=ctx)
nsolver.set_transfer_manager(mg.manifold_transfer_manager(W))

un = fd.Function(V1, name="Velocity")
un.project(case.velocity_expression(*x))
etan = fd.Function(V2, name="Elevation")
etan.project(case.elevation_expression(*x))

u0 = Un.subfunctions[0]
h0 = Un.subfunctions[1]
u0.assign(un)
h0.assign(etan + H - b)
Un1.assign(Un)

Un_ref = fd.Function(W)
Un1_ref = fd.Function(W)

u0_ref, h0_ref = fd.split(Un_ref)
u1_ref, h1_ref = fd.split(Un1_ref)

testeqn_ref = (
    ueq(v, u0_ref, u1_ref, h0_ref, h1_ref)
    + heq(phi, u0_ref, u1_ref, h0_ref, h1_ref)
)


nprob_ref = fd.NonlinearVariationalProblem(testeqn_ref, Un1_ref)
nsolver_ref = fd.NonlinearVariationalSolver(nprob_ref,
                                        solver_parameters=ref_sparams,
                                        appctx=ctx)
nsolver_ref.set_transfer_manager(mg.manifold_transfer_manager(W))

un = fd.Function(V1, name="Velocity")
un.project(case.velocity_expression(*x))
etan = fd.Function(V2, name="Elevation")
etan.project(case.elevation_expression(*x))

u0_ref = Un_ref.subfunctions[0]
h0_ref = Un_ref.subfunctions[1]
u0_ref.assign(un)
h0_ref.assign(etan + H - b)
Un1_ref.assign(Un_ref)

file_sw = fd.File(f'output/{args.filename}.pvd')

itcount = 0
t = 0

from utils.diagnostics import convective_cfl_calculator
cfl_calc = convective_cfl_calculator(mesh)

def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as vec:
        return vec.max()[1]

def save():
    un.assign(u0)
    etan.assign(h0 - H + b)
    file_sw.write(un, etan, time=t)
save()

PETSc.Sys.Print(f"gamma*2/(g*dt) = {args.gamma*2/(earth.gravity*dt)}")

for it in range(args.nt):
    PETSc.Sys.Print(f"\nTimestep: {it} | Hours: {t/units.hour} | CFL: {max_cfl(un, dt)}")

    PETSc.Sys.Print("\nAugmented Lagrangian solver:")
    nsolver.solve()
    Un.assign(Un1)

    PETSc.Sys.Print("\nMonolithic multigrid solver:")
    nsolver_ref.solve()
    Un_ref.assign(Un1_ref)

    PETSc.Sys.Print(f"\nRelative error: {fd.errornorm(Un, Un_ref)/fd.norm(Un)}")

    t += dt
    
    if it % args.save_freq == 0:
        save()

    itcount += nsolver.snes.getLinearSolveIterations()

PETSc.Sys.Print("\nIterations", itcount, "its per step", itcount/args.nt)
