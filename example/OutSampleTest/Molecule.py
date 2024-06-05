import argparse
import os
import sys
import matplotlib.pyplot as plt
import gurobipy as gb
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor

sys.path.append("../..")
from utils.auxiliary_molecule import *
from utils.evolution import out_of_sample_test, obj_of_uncertain_fid

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
# indicator to generate target file
parser.add_argument('--gen_target', help='indicator to generate target file', type=int, default=0)
# file store the target circuit
parser.add_argument('--target', help='unitary matrix of target circuit', type=str, default=None)

# seed for uncertainties
parser.add_argument('--r_seed', help='seed for uncertainties', type=int, default=0)
# group for uncertainties
parser.add_argument('--group', help='group for uncertainties', type=int, default=4)
# number of scenarios
parser.add_argument('--num_scenario', help='number of scenarios', type=int, default=500)
# initial control file
parser.add_argument('--control', help='file name of control', type=str, default=None)
# threshold of CVaR
parser.add_argument('--cvar', help='threshold of CVaR', type=float, default=0.01)
parser.add_argument('--mean', help='mean of Gaussian sample', type=float, default=0)
parser.add_argument('--var', help='variance of Gaussian sample', type=str, default=0.2)
parser.add_argument('--varratio', help='ratio of variance at different time', type=float, default=0.1)
parser.add_argument('--draw', help='indicator whether draw the uncertainty figure', type=int, default=0)
parser.add_argument('--weight', help='weight between expectional and tail objective value', type=float, default=0.5)

args = parser.parse_args()

# args.name="MoleculeNew2"
# args.qubit_num=4
# args.molecule="LiH"
# args.evo_time=20
# args.n_ts=200
# args.target="../control/Continuous/MoleculeNew2_LiH_evotime20.0_n_ts200_target.csv"

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

folder_name = "../../result/output/OutSampleTest/" + args.control.split('/')[-2] + "/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

output_num = folder_name + args.control.split('/')[-1].split('.csv')[0] + "_out_test" + \
             "cvar{}_scenario{}_seed{}_group{}_mean{}_var{}".format(args.cvar, args.num_scenario, args.r_seed,
                                                                    args.group, args.mean, args.var) + ".log"


def test_scenario():
    f = open(output_num, "w+")
    obj = []
    control = np.loadtxt(args.control, delimiter=',')
    num_ctrl = control.shape[1]
    if args.num_scenario == 1:
        c_uncertainty = np.zeros((num_ctrl, args.n_ts, 1))
        obj = np.array(
            out_of_sample_test(args.num_scenario, c_uncertainty, H_d.full(), [h_c.full() for h_c in H_c], args.n_ts,
                               args.evo_time, control, X_0.full(), X_targ, 'fid', true_energy=None))
    else:
        for g in range(args.group):
            seed = args.r_seed + 2 * 2 * args.num_scenario * g * args.n_ts
            if args.var[0] == '[':
                var = args.var[1:-1].split(',')
                var_list = [float(var[0])] * 2 * args.qubit_num + [float(var[1])] * (num_ctrl - 2 * args.qubit_num)
            else:
                var_list = [float(args.var)] * num_ctrl
            c_uncertainty = np.zeros((num_ctrl, args.n_ts, args.num_scenario))
            for i in range(num_ctrl):
                np.random.seed(seed + i * num_ctrl)
                offset = np.random.normal(args.mean, np.sqrt(var_list[i]), size=args.num_scenario)
                time_var = args.varratio * var_list[i]
                for s in range(args.num_scenario):
                    np.random.seed(seed + s * args.n_ts)
                    c_uncertainty[i, :, s] = np.random.normal(offset[s], np.sqrt(time_var), size=args.n_ts)
            prob = np.ones(args.num_scenario) * 1 / args.num_scenario
            obj.append(out_of_sample_test(args.num_scenario, c_uncertainty, H_d.full(), [h_c.full() for h_c in H_c],
                                          args.n_ts, args.evo_time, control, X_0.full(), X_targ, 'fid', None))
        obj = np.concatenate(np.array(obj), axis=0)

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
    print("zeta", zeta_var.x, file=f)
    print("CVaR objective value", m.objval, file=f)
    print("Final maximum objective value {}".format(max(obj)), file=f)
    print("Final average original objective value {}".format(sum(obj) / len(obj)), file=f)
    print("Final original objective value exceeding current objective value {}".format(obj[obj > zeta_var.x]), file=f)
    print("tail objective value {}".format(sum(obj[obj > zeta_var.x] - zeta_var.x) / total_num_scenarios), file=f)
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
    output_log = folder_name + args.control.split('/')[-1].split('.csv')[0] + "_out_test" + \
                 "var{}".format(args.var) + ".log"
    control = np.loadtxt(args.control, delimiter=',')
    comp_name = "../../result/control/Stochastic/MoleculeTest_H2SUR_evotime20.0_n_ts50_ext4000_ptypeCONSTANT_offset0.5_sum_penalty1.0_scenario1_cvar0.05_mean0_var0.001_weight0.5.csv"
    comp_control = np.loadtxt(comp_name, delimiter=",")
    comp_fig = folder_name + comp_name.split('/')[-1].split('.csv')[0] + "_out_test" + \
               "var{}_newlevel".format(args.var) + ".png"

    num_ctrl = control.shape[1]
    if args.var[0] == '[':
        var = args.var[1:-1].split(',')
        var = [float(var[0])] * 2 * args.qubit_num + [float(var[1])] * (num_ctrl - 2 * args.qubit_num)
    else:
        var = [float(args.var)] * num_ctrl
    num_points = 50
    x = np.linspace(-10 * var[0], 10 * var[0], num_points)
    y = np.linspace(-10 * var[1], 10 * var[1], num_points)
    all_obj = []
    all_comp_obj = []
    for ite in range(40):
        uncertainty = np.zeros((num_ctrl, args.n_ts, num_points * num_points))
        totalidx = 0
        for (xidx, xsingle) in enumerate(x):
            for (yidx, ysingle) in enumerate(y):
                # uncertainty[0:2*args.qubit_num, :, totalidx] = np.random.normal(xsingle, np.sqrt(0.1 * var[0]), size=args.n_ts)
                for i in range(4):
                    uncertainty[i, :, totalidx] = np.random.normal(xsingle, np.sqrt(0.1 * var[0]), size=args.n_ts)
                uncertainty[4, :, totalidx] = np.random.normal(ysingle, np.sqrt(0.1 * var[1]), size=args.n_ts)
                # print(xidx, yidx, totalidx)
                totalidx += 1
        obj = out_of_sample_test(num_points * num_points, uncertainty, H_d.full(), [h_c.full() for h_c in H_c],
            args.n_ts, args.evo_time, control, X_0.full(), X_targ, "fid", None)
        comp_obj = out_of_sample_test(num_points * num_points, uncertainty, H_d.full(), [h_c.full() for h_c in H_c],
            args.n_ts, args.evo_time, comp_control, X_0.full(), X_targ, "fid", None)
        f = open(output_log, "a+")
        with np.printoptions(threshold=np.inf):
            print(ite, repr(np.array(obj)), file=f)
        f.close()
        obj_2d = np.array(obj).reshape(num_points, num_points)
        all_obj.append(obj_2d)
        all_comp_obj.append(np.array(comp_obj).reshape(num_points, num_points))
    all_obj = np.array(all_obj)
    for (xidx, xsingle) in enumerate(x):
        for (yidx, ysingle) in enumerate(y):
            print(np.mean(all_obj[:, xidx, yidx]), np.std(all_obj[:, xidx, yidx]), np.min(all_obj[:, xidx, yidx]),
                  np.max(all_obj[:, xidx, yidx]))

    plt.figure(dpi=300)
    cs = plt.contourf(x, y, np.mean(all_obj, axis=0), np.arange(0, 1, 0.01), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap))
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$\mu_1$')
    plt.ylabel(r'$\mu_2$')
    plt.tight_layout()
    plt.savefig(output_fig)

    plt.figure(dpi=300)
    cs = plt.contourf(x, y, np.mean(all_comp_obj, axis=0), np.arange(0, 1, 0.01), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap))
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$\mu_1$')
    plt.ylabel(r'$\mu_2$')
    plt.tight_layout()
    plt.savefig(comp_fig)

    plt.figure(dpi=300)
    ax = plt.axes(projection='3d')
    cs = ax.plot_wireframe(x, y, np.mean(all_obj, axis=0), color='blue', label='Stochastic')
    cs = ax.plot_wireframe(x, y, np.mean(all_comp_obj, axis=0), color='red', label='Deterministic')
    plt.legend()
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$\mu_1$')
    plt.ylabel(r'$\mu_2$')
    plt.tight_layout()
    plt.savefig(output_fig.split(".png")[0] + "_3d_compare.png")

if __name__ == '__main__':
    if args.draw:
        draw_uncertainty()
    else:
        test_scenario()