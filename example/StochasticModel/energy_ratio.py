import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gb

sys.path.append("../..")
from utils.auxiliary_energy import *
from GRAPE import DirectCvarOptimizer
from utils.evolution import compute_TV_norm, out_of_sample_test
from utils.rounding import extend_control, Rounding

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
parser.add_argument('--g_seed', help='graph seed', type=int, default=1)
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
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-4)
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

y0 = uniform(args.n)

if not os.path.exists("../../result/output/Direct/"):
    os.makedirs("../../result/output/Direct/")
if not os.path.exists("../../result/control/Direct/"):
    os.makedirs("../../result/control/Direct/")
if not os.path.exists("../../result/figure/Direct/"):
    os.makedirs("../../result/figure/Direct/")

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

    args.g_seed = 0

if args.ts_b == 0:
    args.ts_b = args.n_ts
if args.binary == 1:
    output_num = "../../result/output/Direct/" + \
                 "{}SUR{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name, str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".log"
else:
    output_num = "../../result/output/Direct/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".log"

output_fig = "../../result/figure/Direct/" + \
             "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                 args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                 args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".png"
output_control = "../../result/control/Direct/" + \
                 "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                     args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.g_seed,
                     args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".csv"

if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.g_seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

# print(Qobj((1 + 0.8) * B).groundstate())

true_energy = min(get_diag(Jij))
c_uncertainty = np.zeros((2, args.n_ts, args.num_scenario))
if args.num_scenario > 1:
    if args.var[0] == '[':
        var = args.var[1:-1].split(',')
        var_list = [float(svar) for svar in var]
    else:
        var_list = [float(args.var)] * 2
    for i in range(2):
        np.random.seed(args.r_seed + i * 2)
        offset = np.random.normal(args.mean, np.sqrt(var_list[i]), size=args.num_scenario)
        time_var = args.varratio * var_list[i]
        for s in range(args.num_scenario):
            np.random.seed(args.r_seed + s * args.n_ts)
            c_uncertainty[i, :, s] = np.random.normal(offset[s], np.sqrt(time_var), size=args.n_ts)

prob = np.ones(args.num_scenario) * 1 / args.num_scenario

if "Seed" in args.name:
    initial_seed = int(args.name.split("Seed")[1])
else:
    initial_seed = 0

if __name__ == '__main__':
    # opt = JointCvarOptimizer()
    opt = DirectCvarOptimizer()
    opt.build_optimizer(None, [B, C], y0[0:2 ** args.n], None, args.n_ts, args.evo_time, max_iter_step=args.max_iter,
                        min_grad=args.min_grad, obj_mode='energyratio', init_type=args.initial_type, constant=args.offset,
                        output_fig=output_fig, output_num=output_num, output_control=output_control, sos1=False,
                        c_uncertainty=c_uncertainty, num_scenario=args.num_scenario, thre_cvar=args.cvar, prob=prob,
                        weight=args.weight, true_energy=min(get_diag(Jij)), seed=initial_seed, initial_control=args.initial_control)

    # control1 = np.array([0] * 20 + [1] * 20)
    # # print(opt._compute_obj(control1))
    # control2 = np.array([1] * 20 + [0] * 20)
    # # print(opt._compute_obj(control2))
    # control2 = 0.4 * control1 + 0.6 * control2
    # for j in range(100):
    #     u = 0.01 * j
    #     control = u * control1 + (1 - u) * control2
    #     print(u, opt._compute_obj(control), u * opt._compute_obj(control1) + (1 - u) * opt._compute_obj(control2))
    # exit()
    opt.optimize_cvar(max_iter=50, lr=0.08)

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
    print("true energy", true_energy, file=f)

    if args.binary == 1:
        b_rel = extend_control(output_control, args.n_ts, args.ts_b, False)
        output_fig = "../../result/figure/Direct/" + \
                     "{}SUR{}_evotime{}_n_ts{}_ext{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                         args.name, str(args.n), args.evo_time, args.n_ts, args.ts_b, args.initial_type, args.offset,
                         args.g_seed, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".png"
        output_control = "../../result/control/Direct/" + \
                         "{}SUR{}_evotime{}_n_ts{}_ext{}_ptype{}_offset{}_instance{}_scenario{}_cvar{}_mean{}_var{}_weight{}".format(
                             args.name, str(args.n), args.evo_time, args.n_ts, args.ts_b, args.initial_type, args.offset,
                             args.g_seed, args.num_scenario, args.cvar, args.mean, args.var, args.weight) + ".csv"
        round = Rounding()
        round.build_rounding_optimizer(b_rel, args.evo_time, args.ts_b, "SUR", out_fig=output_fig)
        b_bin, c_time = round.rounding_with_sos1()
        np.savetxt(output_control, b_bin, delimiter=',')
        exd_c_uncertainty = np.zeros((2, args.ts_b, args.num_scenario))
        step = args.ts_b / args.n_ts
        for t_b in range(args.ts_b):
            con_t = int(np.floor(t_b / step))
            # print(con_t, t_b)
            exd_c_uncertainty[:, t_b, :] = c_uncertainty[:, con_t, :].copy()
        obj = out_of_sample_test(args.num_scenario, exd_c_uncertainty, np.zeros((2 ** args.n, 2 ** args.n), dtype=complex),
                                 [B, C], args.ts_b, args.evo_time, b_bin, y0[0:2 ** args.n], None,
                                 'energyratio', true_energy=true_energy)
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
        print("obtained cvar energy", (1 - m.objval) * true_energy, file=f)
        print("Final maximum objective value {}".format(max(obj)), file=f)
        print("Final original objective value exceeding current objective value {}".format(np.array(obj)[np.array(obj) > m.objval]), file=f)
        f.close()