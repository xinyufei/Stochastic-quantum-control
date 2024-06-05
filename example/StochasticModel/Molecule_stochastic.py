import argparse
import os
import sys
import matplotlib.pyplot as plt
import gurobipy as gb

sys.path.append("../..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from utils.auxiliary_molecule import *
from utils.evolution import compute_TV_norm, out_of_sample_test
from utils.rounding import extend_control, Rounding
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from GRAPE import StochasticCvarOptimizer

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='Molecule')
# name of molecule
parser.add_argument('--molecule', help='molecule name', type=str, default='H2')
# number of quantum bits
parser.add_argument('--qubit_num', help='number of quantum bits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=5)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=100)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# penalty parameter for SOS1 property
parser.add_argument('--sum_penalty', help='penalty parameter for L_2 term', type=float, default=0)
# Fidelity error target
parser.add_argument('--fid_err_targ', help='target for the fidelity error', type=float, default=1e-10)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter', help='maximum number of iterations', type=int, default=300)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time', help='maximum allowed computational time (seconds)', type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-4)
# indicator to generate target file
parser.add_argument('--gen_target', help='indicator to generate target file', type=int, default=0)
# file store the target circuit
parser.add_argument('--target', help='unitary matrix of target circuit', type=str, default=None)

# seed for uncertainties
parser.add_argument('--r_seed', help='seed for uncertainties', type=int, default=0)
# number of scenarios
parser.add_argument('--num_scenario', help='number of scenarios', type=int, default=10)
# threshold of CVaR
parser.add_argument('--cvar', help='threshold of CVaR', type=float, default=0.01)
parser.add_argument('--mean', help='mean of Gaussian sample', type=float, default=0)
parser.add_argument('--var', help='variance of Gaussian sample', type=str, default=0.2)
parser.add_argument('--varratio', help='ratio of variance at different time', type=float, default=0.1)
parser.add_argument('--weight', help='weight between expectional and tail objective value', type=float, default=0.5)

parser.add_argument('--binary', help='indicator if using binary control', type=int, default=0)
parser.add_argument('--ts_b', help='time steps to conduct SUR to obtain binary control', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=0.08)

args = parser.parse_args()

# args.name="MoleculeNew2"
# args.qubit_num=4
# args.molecule="LiH"
# args.evo_time=20
# args.n_ts=200
# args.target="../control/Continuous/MoleculeNew2_LiH_evotime20.0_n_ts200_target.csv"

args.initial_control = "../../result/control/Stochastic/Molecule_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.5_sum_penalty1.0_scenario300_cvar0.05_mean0_var0.001_weight0.5.csv"

d = 2

if args.target is not None:
    Hops, H0, U0, U = generate_molecule_func(args.qubit_num, d, args.molecule, optimize=True, target=args.target)
elif args.gen_target == 1:
    Hops, H0, U0, U = generate_molecule_func(args.qubit_num, d, args.molecule, optimize=True)
else:
    print("Please provide the target file!")
    exit()

# The control Hamiltonians (Qobj classes)
H_c = [Qobj(hops) for hops in Hops]
# Drift Hamiltonian
H_d = Qobj(H0)
# start point for the gate evolution
X_0 = Qobj(U0)
# Target for the gate evolution
X_targ = Qobj(U)

max_controllers = 1
obj_type = "UNIT"

if not os.path.exists("../../result/output/Stochastic/"):
    os.makedirs("../../result/output/Stochastic/")
if not os.path.exists("../../result/control/Stochastic/"):
    os.makedirs("../../result/control/Stochastic")
if not os.path.exists("../../result/figure/Stochastic/"):
    os.makedirs("../../result/figure/Stochastic/")

if args.binary == 1:
    output_num = "../../result/output/Stochastic/" + \
                 "{}SUR_evotime{}_n_ts{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset,
                     args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".log"
    if args.ts_b == 0:
        args.ts_b = args.n_ts
else:
    output_num = "../../result/output/Stochastic/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset,
                     args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".log"

output_fig = "../../result/figure/Stochastic/" + \
             "{}_evotime{}_n_ts{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                 args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset,
                 args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".png"
output_control = "../../result/control/Stochastic/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset,
                     args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".csv"

# solve the optimization model
ops_max_amp = 1
num_ctrl = len(H_c)
var_list = [0.0]
c_uncertainty = np.zeros((num_ctrl, args.n_ts, args.num_scenario))
# if args.num_scenario > 1:
#     if args.var[0] == '[':
#         var = var + args.var[1:-1].split(',')
#     else:
#         var = var + [float(args.var)] * len(H_c)
# print(var)
if args.num_scenario > 1:
    if args.var[0] == '[':
        var = args.var[1:-1].split(',')
        var_list += [float(var[0]) for j in range(2 * args.qubit_num)] + [float(var[1])
                                                                         for j in range(num_ctrl - 2 * args.qubit_num)]
    else:
        var_list += [float(args.var)] * num_ctrl

# print(repr(c_uncertainty))

# exit()

prob = np.ones(args.num_scenario) * 1 / args.num_scenario

if "Seed" in args.name:
    initial_seed = int(args.name.split("Seed")[1])
else:
    initial_seed = 0

if __name__ == '__main__':
    opt = StochasticCvarOptimizer()
    opt.build_optimizer(H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time, max_iter_step=args.max_iter,
                        min_grad=args.min_grad, obj_mode='fid', init_type=args.initial_type, initial_control=args.initial_control,
                        constant=args.offset, output_fig=output_fig, output_num=output_num, penalty=args.sum_penalty, varratio=args.varratio,
                        output_control=output_control, sos1=True, sample_size=args.num_scenario, var=var_list, thre_cvar=args.cvar,
                        weight=args.weight, true_energy=None, seed=initial_seed)
    opt.optimize_cvar(args.max_iter, args.lr)

    b_rel = np.loadtxt(output_control, delimiter=",")
    if len(b_rel.shape) == 1:
        b_rel = np.expand_dims(b_rel, axis=1)
    fig = plt.figure(dpi=300)
    # plt.title("Optimised Quantum Control Sequences")
    plt.xlabel("Time")
    plt.ylabel("Control amplitude")
    plt.ylim([0, 1])
    marker_list = ['-o', '--^', '-*', '--s', '--+']
    marker_size_list = [5, 5, 8, 5, 8]
    for j in range(b_rel.shape[1]):
        plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j % 5],
                 where='post', linewidth=2, label='controller ' + str((j + 1) % 5), markevery=(j, 4),
                 markersize=marker_size_list[j % 5])
    plt.legend()
    plt.savefig(output_fig.split(".png")[0] + "_continuous.png")

    f = open(output_num, "a+")
    print("total tv norm", compute_TV_norm(b_rel), file=f)

    if args.binary == 1:
        b_rel = extend_control(output_control, args.n_ts, args.ts_b, False)
        output_fig = "../../result/figure/Stochastic/" + \
                     "{}SUR_evotime{}_n_ts{}_ext{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                         args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.ts_b, args.initial_type,
                         args.offset, args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".png"
        output_control = "../../result/control/Stochastic/" + \
                         "{}SUR_evotime{}_n_ts{}_ext{}_ptype{}_offset{}_sum_penalty{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                         args.name + "Sto_" + args.molecule, args.evo_time, args.n_ts, args.ts_b, args.initial_type, args.offset,
                         args.sum_penalty, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".csv"
        round = Rounding()
        round.build_rounding_optimizer(b_rel, args.evo_time, args.ts_b, "SUR", out_fig=output_fig)
        b_bin, c_time = round.rounding_with_sos1()
        np.savetxt(output_control, b_bin, delimiter=',')
        exd_c_uncertainty = np.zeros((num_ctrl, args.ts_b, args.num_scenario))
        step = args.ts_b / args.n_ts
        for t_b in range(args.ts_b):
            con_t = int(np.floor(t_b / step))
            exd_c_uncertainty[:, t_b, :] = c_uncertainty[:, con_t, :].copy()
        obj = out_of_sample_test(args.num_scenario, exd_c_uncertainty, H_d.full(), [h_c.full() for h_c in H_c],
                                 args.ts_b, args.evo_time, b_bin, X_0.full(), X_targ, 'fid', true_energy=None)
        m = gb.Model()
        zeta_var = m.addVar(lb=-np.infty)
        cos = m.addVars(args.num_scenario, lb=0)
        m.addConstrs(cos[l] >= obj[l] - zeta_var for l in range(args.num_scenario))
        m.setObjective(zeta_var + 1 / args.cvar * gb.quicksum(prob[l] * cos[l] for l in range(args.num_scenario)))
        m.optimize()
        f = open(output_num, "a+")
        print("********************Rounding results********************", file=f)
        print("extended time steps", args.ts_b, file=f)
        print("rounding time", c_time, file=f)
        ave_obj = sum(prob[l] * obj[l] for l in range(args.num_scenario))
        print("Final total objective value", ave_obj * args.weight + m.objval * (1 - args.weight), file=f)
        print("Final average objective value", ave_obj, file=f)
        print("Final cvar objective value", m.objval, file=f)
        print("zeta", zeta_var.x, file=f)
        print("Final maximum objective value {}".format(max(obj)), file=f)
        print("Final original objective value exceeding current objective value {}".format(
            np.array(obj)[np.array(obj) > m.objval]), file=f)
        f.close()