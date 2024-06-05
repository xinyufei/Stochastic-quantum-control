import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gb

sys.path.append("../..")
from utils.auxiliary_energy import *
from utils.evolution import out_of_sample_test, obj_of_uncertain

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
# group for uncertainties
parser.add_argument('--group', help='group for uncertainties', type=int, default=6)
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
parser.add_argument('--mean', help='mean of Gaussian sample', type=float, default=0)
parser.add_argument('--var', help='variance of Gaussian sample', type=str, default=0.2)
parser.add_argument('--varratio', help='ratio of variance at different time', type=float, default=0.1)
parser.add_argument('--draw', help='indicator whether draw the uncertainty figure', type=int, default=0)

args = parser.parse_args()

y0 = uniform(args.n)

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

    args.seed = 0

if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.g_seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

true_energy = min(get_diag(Jij))


def test_scenario():
    folder_name = "../../result/output/OutSampleTest/" + args.control.split('/')[-2] + "/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_num = folder_name + args.control.split('/')[-1].split('.csv')[0] + "_out_test" + \
                 "cvar{}_scenario{}_seed{}_group{}_mean{}_var{}".format(args.cvar, args.num_scenario, args.r_seed,
                                                                        args.group, args.mean, args.var) + ".log"
    f = open(output_num, "w+")
    obj = []
    control = np.loadtxt(args.control, delimiter=',')
    if args.num_scenario == 1:
        c_uncertainty = np.zeros((2, 1))
        obj = out_of_sample_test(args.num_scenario, c_uncertainty, np.zeros((2 ** args.n, 2 ** args.n), dtype=complex),
                                 [B, C], args.n_ts, args.evo_time, control, y0[0:2 ** args.n], None, 'energyratio',
                                 true_energy=true_energy)
    else:
        for g in range(args.group):
            seed = args.r_seed + 2 * 2 * args.num_scenario * g * args.n_ts
            if args.var[0] == '[':
                var = args.var[1:-1].split(',')
                var_list = [float(svar) for svar in var]
            else:
                var_list = [float(args.var)] * 2
            c_uncertainty = np.zeros((2, args.n_ts, args.num_scenario))
            for i in range(2):
                np.random.seed(seed + i * 2)
                offset = np.random.normal(args.mean, np.sqrt(var_list[i]), size=args.num_scenario)
                time_var = args.varratio * var_list[i]
                for s in range(args.num_scenario):
                    np.random.seed(seed + s * args.n_ts)
                    c_uncertainty[i, :, s] = np.random.normal(offset[s], np.sqrt(time_var), size=args.n_ts)
            # if args.var[0] == '[':
            #     var = args.var[1:-1].split(',')
            #     c_uncertainty = np.zeros((2, args.num_scenario))
            #     for i in range(2):
            #         cur_seed = seed + 2 * i
            #         np.random.seed(cur_seed)
            #         c_uncertainty[i, :] = np.random.normal(args.mean, np.sqrt(float(var[i])), size=args.num_scenario)
            # else:
            #     np.random.seed(seed)
            #     # c_uncertainty = np.random.uniform(low=-args.bound, high=args.bound, size=(2, args.num_scenario))
            #     c_uncertainty = np.random.normal(args.mean, np.sqrt(float(args.var)), size=(2, args.num_scenario))
            prob = np.ones(args.num_scenario) * 1 / args.num_scenario
            obj.append(out_of_sample_test(
                args.num_scenario, c_uncertainty, np.zeros((2 ** args.n, 2 ** args.n), dtype=complex),
                [B, C], args.n_ts, args.evo_time, control, y0[0:2 ** args.n], None, 'energyratio',
                true_energy=true_energy))
        obj = np.concatenate(np.array(obj), axis=0)

    diag = get_diag(Jij)
    diag.sort()
    first_excited_energy = 1 - diag[2] / diag[0]

    total_num_scenarios = len(obj)
    prob = np.ones(total_num_scenarios) * 1 / total_num_scenarios

    m = gb.Model()
    zeta_var = m.addVar(lb=-np.infty)
    cos = m.addVars(total_num_scenarios, lb=0)
    m.addConstrs(cos[l] >= obj[l] - zeta_var for l in range(total_num_scenarios))
    m.setObjective(zeta_var + 1 / args.cvar * gb.quicksum(prob[l] * cos[l] for l in range(total_num_scenarios)))
    m.optimize()

    print("********* Summary *****************", file=f)
    print("start seed", args.r_seed, file=f)
    print("true energy and first excited energy", diag[0], diag[2], file=f)
    print("zeta", zeta_var.x, file=f)
    print("CVaR objective value", m.objval, file=f)
    print("Final maximum objective value {}".format(max(obj)), file=f)
    print("Final average original objective value {}".format(sum(obj) / len(obj)), file=f)
    print("Final original objective value exceeding current objective value {}".format(obj[obj > zeta_var.x]), file=f)
    print("tail objective value {}".format(sum(obj[obj > zeta_var.x] - zeta_var.x) / total_num_scenarios), file=f)
    print("Distinguished percentage {}".format(len(obj[obj < first_excited_energy]) / (args.group * args.num_scenario)),
          file=f)
    print("95% percentile is {}".format(np.percentile(obj, 95)), file=f)
    print("********* Results *****************", file=f)
    print("Final original objective value", file=f)
    for l in range(total_num_scenarios):
        print(obj[l], file=f)


def draw_uncertainty():
    folder_name = "../../result/figure/OutSampleTest/" + args.control.split('/')[-2] + "/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_fig = folder_name + args.control.split('/')[-1].split('.csv')[0] + "_out_test" + \
                 "var{}_newlevel".format(args.var) + ".png"
    control = np.loadtxt(args.control, delimiter=',')

    if args.var[0] == '[':
        var = args.var[1:-1].split(',')
    else:
        var = [float(args.var)] * 2
    num_points = 50
    x = np.linspace(-10 * var[0], 10 * var[0], num_points)
    y = np.linspace(-10 * var[1], 10 * var[1], num_points)
    obj = obj_of_uncertain(np.meshgrid(x, y), np.zeros((2 ** args.n, 2 ** args.n), dtype=complex),
                [B, C], args.n_ts, args.evo_time, control, y0[0:2 ** args.n], None, 'energyratio',
                true_energy=true_energy)
    obj_2d = np.array(obj).reshape(num_points, num_points).T
    # i = 0
    # for (epsilon_0, epsilon_1) in zip(*(sample.flat for sample in np.meshgrid(x, y))):
    #     print(epsilon_0, epsilon_1, obj[i])
    #     i += 1
    print(np.max(obj_2d))
    plt.figure(dpi=300)
    cs = plt.contourf(x, y, obj_2d, np.arange(0, 1, 0.01), cmap='viridis', vmin=0, vmax=0.45)
    plt.colorbar(plt.cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap))
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$\epsilon_1$')
    plt.ylabel(r'$\epsilon_2$')
    plt.tight_layout()
    plt.savefig(output_fig)


if __name__ == '__main__':
    if args.draw:
        draw_uncertainty()
    else:
        test_scenario()