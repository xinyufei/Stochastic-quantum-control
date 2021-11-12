import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gb

sys.path.append("../..")
from utils.auxiliary_energy import *
from utils.evolution import out_of_sample_test

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
parser.add_argument('--num_scenario', help='number of scenarios', type=int, default=100)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=2)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=40)
# initial control file
parser.add_argument('--control', help='file name of control', type=str, default=None)
# threshold of CVaR
parser.add_argument('--cvar', help='threshold of CVaR', type=float, default=0.01)

args = parser.parse_args()

y0 = uniform(args.n)

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

    args.seed = 0

if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

if args.num_scenario == 1:
    c_uncertainty = np.zeros((2, 1))
else:
    np.random.seed(args.r_seed)
    c_uncertainty = np.random.uniform(low=-0.2, high=0.2, size=(2, args.num_scenario))
prob = np.ones(args.num_scenario) * 1 / args.num_scenario

if not os.path.exists("../../output/OutSampleTest/"):
    os.makedirs("../../output/OutSampleTest/")
if not os.path.exists("../../control/OutSampleTest/"):
    os.makedirs("../../control/OutSampleTest/")
if not os.path.exists("../../figure/OutSampleTest/"):
    os.makedirs("../../figure/OutSampleTest/")

# args.control = "../../control/Stepcvar/Energy2_evotime2_n_ts40_ptypeCONSTANT_offset0.5_instance0_scenario10_cvar0.01.csv"
args.control = "../../control/Average/Energy2_evotime2_n_ts40_ptypeCONSTANT_offset0.5_instance0.csv"

output_num = "../../output/OutSampleTest/" + args.control.split('/')[-1].split('.csv')[0] + "out_test" + \
             "_scenario{}".format(args.num_scenario) + ".log"

f = open(output_num, "a+")
control = np.loadtxt(args.control, delimiter=',')
obj = out_of_sample_test(args.num_scenario, c_uncertainty, np.zeros((2 ** args.n, 2 ** args.n), dtype=complex),
                         [B, C], args.n_ts, args.evo_time, control, y0[0:2 ** args.n], None, 'energy')
print("********* Results *****************", file=f)
m = gb.Model()
zeta_var = m.addVar(lb=-np.infty)
cos = m.addVars(args.num_scenario, lb=0)
m.addConstrs(cos[l] >= obj[l] - zeta_var for l in range(args.num_scenario))
m.setObjective(zeta_var + 1 / args.cvar * gb.quicksum(prob[l] * cos[l] for l in range(args.num_scenario)))
m.optimize()
print("zeta", zeta_var.x, file=f)
print("objective value", m.objval, file=f)
print("Final original objective value {}".format(obj), file=f)
print("Final maximum energy {}".format(-min(abs(obj))), file=f)
print("Final average original objective value {}".format(sum(obj) / len(obj)), file=f)
print("Final original objective value exceeding current objective value {}".format(obj[obj > m.objval]), file=f)
