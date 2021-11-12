import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gb
sys.path.append("../..")
from utils.auxiliary_energy import *
from GRAPE import JointCvarOptimizer, StepCvarOptimizer
from utils.evolution import compute_TV_norm, time_evolution, compute_obj_energy

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='Energy')
# number of qubits
parser.add_argument('--n', help='number of qubits', type=int, default=2)
# number of edges for generating regular graph
parser.add_argument('--num_edges', help='number of edges for generating regular graph', type=int,
                    default=1)
# if generate the graph randomly
parser.add_argument('--rgraph', help='if generate the graph randomly', type=int, default=0)
# number of instances
parser.add_argument('--g_seed', help='graph seed', type=int, default=0)
# seed for uncertainties
parser.add_argument('--r_seed', help='seed for uncertainties', type=int, default=0)
# number of scenarios
parser.add_argument('--num_scenario', help='number of scenarios', type=int, default=10)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=2)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=40)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter', help='maximum number of iterations', type=int, default=100)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-6)
# threshold of CVaR
parser.add_argument('--cvar', help='threshold of CVaR', type=float, default=0.01)
parser.add_argument('--bound', help='bound of uniform sample', type=float, default=0.2)

args = parser.parse_args()

y0 = uniform(args.n)

if not os.path.exists("../../output/Stepcvar/"):
    os.makedirs("../../output/Stepcvar/")
if not os.path.exists("../../control/Stepcvar/"):
    os.makedirs("../../control/Stepcvar/")
if not os.path.exists("../../figure/Stepcvar/"):
    os.makedirs("../../figure/Stepcvar/")

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)
    
    args.seed = 0
    

output_num = "../../output/Stepcvar/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_bound{}".format(
                     args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.bound) + ".log"
output_fig = "../../figure/Stepcvar/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_bound{}".format(
                     args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.bound) + ".png"
output_control = "../../control/Stepcvar/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_bound{}".format(
                     args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.bound) + ".csv"
    
if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.g_seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

if args.num_scenario == 1:
    c_uncertainty = np.zeros((2, 1))
else:
    np.random.seed(args.r_seed)
    c_uncertainty = np.random.uniform(low=-args.bound, high=args.bound, size=(2, args.num_scenario))
prob = np.ones(args.num_scenario) * 1 / args.num_scenario

# opt = JointCvarOptimizer()
opt = StepCvarOptimizer()
opt.build_optimizer(None, [B, C], y0[0:2**args.n], None, args.n_ts, args.evo_time, max_iter_step=args.max_iter,
                    min_grad=args.min_grad, obj_mode='energy', init_type=args.initial_type, constant=args.offset,
                    output_fig=output_fig, output_num=output_num, output_control=output_control, sos1=False,
                    c_uncertainty=c_uncertainty, num_scenario=args.num_scenario, thre_cvar=args.cvar, prob=prob)

opt.optimize_cvar()

b_rel = np.loadtxt(output_control, delimiter=",")
if len(b_rel.shape) == 1:
    b_rel = np.expand_dims(b_rel, axis=1)
fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.ylim([0, 1])
marker_list = ['-o', '--^', '-*', '--s']
marker_size_list = [5, 5, 8, 5]
for j in range(b_rel.shape[1]):
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig(output_fig.split(".png")[0] + "_continuous.png")

f = open(output_num, "a+")
print("total tv norm", compute_TV_norm(b_rel), file=f)
print("true energy", sum(prob[l] * (1 + c_uncertainty[1, l]) * min(get_diag(Jij)) for l in range(args.num_scenario)),
      file=f)
