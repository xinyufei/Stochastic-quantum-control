import math
from qutip.control.optimizer import *

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import qutip.control.pulseoptim as cpo
from qutip import Qobj
import sys

import scipy

from scipy.linalg import expm
import scipy.optimize
import gurobipy as gb
import multiprocessing
from utils.evolution import out_of_sample_test


def time_evolution_energy_each_scenario(init_state, c_uncertainty, c_hamil, control_amps, delta_t):
    into = [init_state]
    for k in range(control_amps.shape[0]):
        fwd = expm(-1j * (control_amps[k] * (1 + c_uncertainty[0, k]) * c_hamil[0] +
                          (1 - control_amps[k]) * (1 + c_uncertainty[1, k]) * c_hamil[1]) * delta_t).dot(into[k])
        into.append(fwd)
    return into


def back_propagation_energy_each_scenario(final_state, c_uncertainty, control_amps, delta_t, c_hamil):
    onto = [final_state.conj().T.dot(c_hamil[1].conj().T)]
    n_ts = control_amps.shape[0]
    for k in range(n_ts):
        bwd = onto[k].dot(expm(-1j * (control_amps[n_ts - k - 1] * (1 + c_uncertainty[0, n_ts - k - 1]) * c_hamil[0] +
                                      (1 - control_amps[n_ts - k - 1]) * (1 + c_uncertainty[1, n_ts - k - 1])
                                      * c_hamil[1]) * delta_t))
        onto.append(bwd)
    return onto


def compute_fid_obj(optim, ctrl):
    return optim.fid_err_func_compute(ctrl)


def compute_fid_gradient(optim, ctrl):
    return optim.fid_err_grad_compute(ctrl)


def compute_energy_gradient(onto, into, c_uncertainty, c_hamil, n_ts, delta_t):
    grad = []
    for k in range(n_ts):
        cur_grad = np.imag(onto[n_ts - k - 1].dot(((1 + c_uncertainty[0, k]) * c_hamil[0] -
                                                   (1 + c_uncertainty[1, k]) * c_hamil[1]).dot(into[k + 1]))
                           * delta_t) * 2
        grad += [cur_grad]
    return np.expand_dims(np.array(grad), 1)


class StochasticCvarOptimizer:
    """
    optimal controller for CVaR robust model
    """

    def __init__(self):
        self.d_hamil = None
        self.c_hamil = None
        self.d_hamil_qobj = None
        self.c_hamil_qobj = None
        self.init_state = None
        self.targ_state = None
        self.init_state_qobj = None
        self.targ_state_qobj = None
        self.n_ts = 0
        self.evo_time = None
        self.amp_lbound = None
        self.amp_ubound = None
        self.ops_max_amp = 1
        self.fid_err_targ = None
        self.min_grad = None
        self.max_iter_step = None
        self.max_wall_time_step = None
        self.obj_mode = "fid"
        self.init_type = None
        self.seed = None
        self.constant = None
        self.initial_control = None
        self.output_num = None
        self.output_fig = None
        self.output_control = None
        self.sum_cons_1 = False
        self.n_ctrls = None
        self.admm_err_targ = None
        self.time_optimize_start_step = 0
        self.num_iter_step = 0
        self.penalty = np.infty
        self.d_uncertain = None
        self.c_uncertain = None
        self.num_scenario = None
        self.thre_cvar = 0
        self.prob = None

        self.obj = []
        self.cur_obj = 0
        self._into = []
        self._onto = []
        self.delta_t = 0

        self.u = None
        self.zeta = 0
        self.termination_reason = None

        self.result = None
        self.optim = []

        self.true_energy = []
        self.mean_true_energy = None
        self.r_d_hamil = None
        self.r_c_hamil = None
        self.pool = None

    def build_optimizer(self, d_hamil, c_hamil, init_state, targ_state, n_ts, evo_time, amp_lbound=0, amp_ubound=1,
                        min_grad=1e-6, max_iter_step=500, obj_mode="fid", init_type="ZERO", seed=None, constant=0,
                        initial_control=None, output_num=None, output_fig=None, output_control=None,
                        sos1=False, penalty=10, sample_size=40, var=np.ones(5) * 0.01, varratio=0.1,
                        thre_cvar=0, weight=0, true_energy=None):
        self.d_hamil = d_hamil
        self.c_hamil = c_hamil
        self.init_state = init_state
        self.targ_state = targ_state
        self.n_ts = n_ts
        self.evo_time = evo_time
        self.amp_lbound = amp_lbound
        self.amp_ubound = amp_ubound
        self.min_grad = min_grad
        self.max_iter_step = max_iter_step
        self.obj_mode = obj_mode
        self.init_type = init_type
        self.constant = constant
        self.initial_control = initial_control
        self.output_num = output_num
        self.output_fig = output_fig
        self.output_control = output_control
        self.seed = seed
        self.penalty = penalty
        self.sum_cons_1 = sos1
        self.thre_cvar = thre_cvar
        self.var = var
        self.varratio = varratio
        self.weight = weight
        self.sample_size = sample_size

        self.delta_t = evo_time / n_ts
        self.r_d_hamil = [None] * self.sample_size
        self.r_c_hamil = [None] * self.sample_size

        if self.obj_mode == 'fid':
            self.d_hamil_qobj = Qobj(d_hamil)
            self.c_hamil_qobj = [Qobj(c_hamil_j) for c_hamil_j in c_hamil]
            self.init_state_qobj = Qobj(init_state)
            self.targ_state_qobj = Qobj(targ_state)
            self.n_ctrls = len(self.c_hamil)
            self.u = np.zeros((self.n_ts, self.n_ctrls))

        if self.obj_mode in ['energy', 'energyratio']:
            self.n_ctrls = len(self.c_hamil) - 1
            self._into = [None] * self.sample_size
            self._onto = [None] * self.sample_size
            self.u = np.zeros(self.n_ts)
            self.mean_true_energy = true_energy

        self.pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        self.obj = np.zeros(self.sample_size)

    def _initialize_control(self):
        """
        :param self:
        :return: an n_ts*n_ctrls array
        """
        self.init_amps = np.zeros([self.n_ts, self.n_ctrls])
        if self.init_type == "RND":
            if self.seed:
                np.random.seed(self.seed)
            self.init_amps = np.random.rand(
                self.n_ts, self.n_ctrls) * (self.amp_ubound - self.amp_lbound) + self.amp_lbound
        if self.init_type == "CONSTANT":
            self.init_amps = np.zeros((self.n_ts, self.n_ctrls)) + self.constant
        if self.init_type == "WARM":
            # file = open(self.initial_control)
            warm_start_control = np.loadtxt(self.initial_control, delimiter=",")
            evo_time_start = warm_start_control.shape[0]
            step = self.n_ts / evo_time_start
            for j in range(self.n_ctrls):
                for time_step in range(self.n_ts):
                    self.init_amps[time_step, j] = warm_start_control[int(np.floor(time_step / step)), j]
        self.init_amps /= self.init_amps.sum(axis=1, keepdims=True)


    def _time_evolution_energy(self, control_amps):
        self._into = self.pool.starmap(time_evolution_energy_each_scenario,
                                       [(self.init_state, self.c_uncertain[:, :, l], self.c_hamil, control_amps,
                                         self.delta_t) for l in range(self.sample_size)])

    def _back_propagation_energy(self, control_amps):
        self._onto = self.pool.starmap(back_propagation_energy_each_scenario,
                                       [(self._into[l][-1], self.c_uncertain[:, :, l],
                                         control_amps, self.delta_t, self.c_hamil) for l in range(self.sample_size)])

    def _compute_origin_obj(self, control_amps, recompute=False):
        change = np.array_equal(self.u, control_amps)
        if not recompute and change:
            return self.iter_obj
        self.obj = np.zeros(self.sample_size)
        reshape_ctrl = control_amps.reshape(self.n_ts, self.n_ctrls)
        # print(control_amps.reshape(self.n_ts, self.n_ctrls))
        if self.obj_mode == 'fid':
            self.obj = np.array(self.pool.starmap(compute_fid_obj, [(self.optim[l], reshape_ctrl)
                                                                    for l in range(self.sample_size)]))
        if self.obj_mode in ['energy', 'energyratio']:
            if not (self.u == control_amps).all():
                self._time_evolution_energy(control_amps)
                self._back_propagation_energy(control_amps)
                self.u = control_amps
                if self.obj_mode == "energy":
                    self.obj = [np.real(self._into[idx][-1].conj().T.dot(self.c_hamil[1].dot(self._into[idx][-1])))
                                for idx in range(self.sample_size)]
                if self.obj_mode == "energyratio":
                    for l in range(self.sample_size):
                        self.obj = np.array(
                            [1 - np.real(self._into[idx][-1].conj().T.dot(self.c_hamil[1].dot(self._into[idx][-1])))
                             / self.mean_true_energy for idx in range(self.sample_size)])
        self.u = control_amps

    def _compute_l2_penalty(self, control_amps):
        penalty = sum(np.power(sum(control_amps[t, j] for j in range(self.n_ctrls)) - 1, 2) for t in range(self.n_ts))
        return penalty

    def _compute_obj(self, *args):
        """
        :param args: control list
        :return: error
        """
        control_amps = args[0].copy()
        self._compute_origin_obj(*args)
        self.u = control_amps
        # directly compute zeta
        if len(self.obj) == 0:
            self.zeta = -1
            self.l_star = -1
            sorted_obj = self.obj
        else:
            sorted_obj = np.sort(self.obj)
            self.l_star = self.sample_size - 1 - math.ceil(self.thre_cvar * self.sample_size)
            self.zeta = sorted_obj[self.l_star]
        self.cvar_obj = self.zeta + 1 / self.thre_cvar * sum(
            1 / self.sample_size * (sorted_obj[l] - self.zeta) for l in range(self.l_star + 1, self.sample_size))
        self.ave_obj = sum(self.obj) / len(self.obj)
        obj = self.weight * self.ave_obj + (1 - self.weight) * self.cvar_obj
        penalized = 0
        if self.sum_cons_1:
            penalized = self.penalty * self._compute_l2_penalty(control_amps.reshape([self.n_ts, self.n_ctrls]))
        self.iter_obj = obj + penalized
        return self.iter_obj

    def _compute_gradient(self, *args):
        control_amps = args[0].copy()
        cvar_grad_control = np.zeros((self.n_ts, self.n_ctrls))
        ave_grad_control = np.zeros((self.n_ts, self.n_ctrls))
        penalized_grad = np.zeros((self.n_ts, self.n_ctrls))

        # print(control_amps)
        if self.obj_mode == 'fid':
            change = np.array_equal(self.u, control_amps)
        else:
            change = (self.u == control_amps).all()
        if not change:
            obj = self._compute_obj(*args)

        exceed_idx = []

        reshape_control = np.reshape(control_amps, (self.n_ts, self.n_ctrls))
        if self.obj_mode == 'fid':
            # all_grad = self.pool.starmap(compute_fid_gradient, [(self.optim[l], reshape_control)
            #                                                     for l in range(self.sample_size)])
            all_grad = self.pool.starmap(compute_fid_gradient, [(self.optim[l], reshape_control)
                                                                for l in range(self.sample_size)])
            for idx in range(self.sample_size):
                # grad = self.optim[l].fid_err_grad_compute(reshape_control)
                grad = all_grad[idx]
                if self.zeta < self.obj[idx]:
                    # cvar_grad_control += self.prob[l] / self.thre_cvar * grad
                    cvar_grad_control += 1 / self.sample_size / self.thre_cvar * grad
                    exceed_idx.append(idx)
                ave_grad_control += 1 / self.sample_size * grad

            if self.sum_cons_1:
                for t in range(self.n_ts):
                    grad_p = 2 * self.penalty * (sum(reshape_control[t, j] for j in range(self.n_ctrls)) - 1)
                    for j in range(self.n_ctrls):
                        penalized_grad[t, j] = grad_p

        if self.obj_mode == 'energy':
            all_grad = self.pool.starmap(
                compute_energy_gradient, [(self._onto[l], self._into[l], self.c_uncertain[:, :, l],
                                           self.c_hamil, self.n_ts, self.delta_t)
                                          for l in range(self.sample_size)])
            for idx in range(self.sample_size):
                if self.zeta < self.obj[idx]:
                    cvar_grad_control += 1 / self.sample_size / self.thre_cvar * all_grad[idx]
                    exceed_idx.append(idx)
                ave_grad_control += 1 / self.sample_size * all_grad[idx]

        if self.obj_mode == 'energyratio':
            all_grad = self.pool.starmap(
                compute_energy_gradient, [(self._onto[l], self._into[l], self.c_uncertain[:, :, l],
                                           self.c_hamil, self.n_ts, self.delta_t)
                                          for l in range(self.sample_size)])

            for idx in range(self.sample_size):
                if self.zeta < self.obj[idx]:
                    cvar_grad_control += 1 / self.sample_size / self.thre_cvar * all_grad[idx] / (
                        -self.mean_true_energy)
                    exceed_idx.append(idx)
                ave_grad_control += 1 / self.sample_size * all_grad[idx]

        # l_star_coeff = sum(self.prob[l] for l in exceed_idx) / self.thre_cvar
        l_star_coeff = 1 / self.sample_size * len(exceed_idx) / self.thre_cvar
        # print(l_star_coeff)
        if l_star_coeff != 1:
            cvar_grad_control += (1 - l_star_coeff) * all_grad[self.l_star]
        grad_control = self.weight * ave_grad_control + (1 - self.weight) * cvar_grad_control
        if self.obj_mode == "energyratio":
            grad_control /= -self.mean_true_energy
        # print(grad_control.flatten() + penalized_grad.flatten())

        return grad_control.flatten() + penalized_grad.flatten()

    def _adam(self, initial_amps, max_iter, learning_rate, test=False):
        self.learning_rate_large = learning_rate
        self.learning_rate_small = 0.1 * self.learning_rate_large
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        x = initial_amps
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        self.obj_list = []
        self.convert_iter = 1000
        learning_rate = self.learning_rate_large
        start = time.time()
        for ite in range(max_iter):
            if self.obj_mode == "fid":
                self.d_uncertain = np.zeros((self.n_ts, self.sample_size))
                d_offset = np.random.normal(0, np.sqrt(self.var[0]), size=self.sample_size)
                self.c_uncertain = np.zeros((self.n_ctrls, self.n_ts, self.sample_size))
                c_offset = np.zeros((self.n_ctrls, self.sample_size))
                for j in range(self.n_ctrls):
                    c_offset[j, :] = np.random.normal(0, np.sqrt(self.var[j + 1]), size=self.sample_size)
                self.optim = []
                for l in range(self.sample_size):
                    self.d_uncertain[:, l] = np.random.normal(d_offset[l], np.sqrt(self.var[0] * self.varratio),
                                                         size=self.n_ts)
                    self.r_d_hamil[l] = [(1 + self.d_uncertain[k, l]) * self.d_hamil_qobj for k in range(self.n_ts)]
                    for j in range(self.n_ctrls):
                        self.c_uncertain[j, :, l] = np.random.normal(c_offset[j, l],
                                                                np.sqrt(self.var[j + 1] * self.varratio),
                                                                size=self.n_ts)
                    self.r_c_hamil[l] = [[(1 + self.c_uncertain[j, k, l]) * self.c_hamil_qobj[j]
                                          for j in range(self.n_ctrls)] for k in range(self.n_ts)]
                    self.optim.append(
                        cpo.create_pulse_optimizer(self.r_d_hamil[l], self.r_c_hamil[l], self.init_state_qobj,
                                                   self.targ_state_qobj, self.n_ts, self.evo_time,
                                                   amp_lbound=self.amp_lbound, amp_ubound=self.amp_ubound,
                                                   dyn_type='UNIT', fid_params={'phase_option': "PSU"},
                                                   init_pulse_params={"offset": 0}, gen_stats=True))
                    dyn = self.optim[l].dynamics
                    dyn.initialize_controls(self.init_amps)
            if self.obj_mode in ["energy", "energyratio"]:
                self.c_uncertain = np.zeros((2, self.n_ts, self.sample_size))
                c_offset = np.zeros((2, self.sample_size))
                for j in range(2):
                    c_offset[j, :] = np.random.normal(0, np.sqrt(self.var[j]), size=self.sample_size)
                for l in range(self.sample_size):
                    for j in range(2):
                        self.c_uncertain[j, :, l] = np.random.normal(
                            c_offset[j, l], np.sqrt(self.varratio * self.var[j]), size=self.n_ts)
            g = self._compute_gradient(x)
            # ubd_g = x * (1 - x) * g

            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g ** 2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1 ** (ite + 1))  # bias correction.
            vhat = v / (1 - beta2 ** (ite + 1))

            pre_x = x.copy()
            x = np.clip(x - learning_rate * mhat / (np.sqrt(vhat) + eps), 0, 1)

            obj = self._compute_obj(x)
            report = open(self.output_num, "a+")
            print("iteration", ite + 1, obj, file=report)
            # print("iteration", ite + 1, obj)
            self.obj_list.append(obj)

            if test:
                c_uncertain = np.zeros((c_uncertain.shape[0], 2000))
                for i in range(c_uncertain.shape[0]):
                    c_uncertain[i, :] = np.random.normal(0, np.sqrt(self.var[i + 1]), size=2000)
                if self.obj_mode == "fid":
                    reshapex = np.reshape(x, (self.n_ts, self.n_ctrls))
                    test_obj_list = out_of_sample_test(2000, c_uncertain, self.d_hamil, self.c_hamil,
                                                       self.n_ts, self.evo_time, reshapex,
                                                       self.init_state, self.targ_state, 'fid', None, False)
                if self.obj_mode in ["energy", "energyratio"]:
                    reshapex = np.reshape(x, (self.n_ts, 1))
                    test_obj_list = out_of_sample_test(2000, c_uncertain,
                                                       np.zeros((self.c_hamil[0].shape[0], self.c_hamil[0].shape[1]),
                                                                dtype=complex),
                                                       self.c_hamil, self.n_ts, self.evo_time,
                                                       np.concatenate(reshapex, 1 - reshapex),
                                                       self.init_state, None, self.obj_mode,
                                                       true_energy=self.mean_true_energy, binary=False)

                eva = gb.Model()
                zeta_var = eva.addVar(lb=-np.infty)
                cos = eva.addVars(2000, lb=0)
                eva.addConstrs(cos[l] >= test_obj_list[l] - zeta_var for l in range(2000))
                eva.setObjective(gb.quicksum(1 / 2000 * (self.weight * test_obj_list[l]) + (1 - self.weight) * cos[l]
                                             for l in range(2000)) + (1 - self.weight) * zeta_var)
                eva.Params.LogToConsole = 0
                eva.optimize()

                test_obj = eva.objval

                if self.obj_mode == "fid" and self.sum_cons_1:
                    test_obj += self.penalty * self._compute_l2_penalty(reshapex)

                print("iteration", ite + 1, "test", test_obj, file=report)
                # print("iteration", ite + 1, "test", test_obj)
            report.close()
            # if ite == self.convert_iter and obj < np.sqrt():
            #     learning_rate = self.learning_rate_small
            if obj < 0.1:
                learning_rate = self.learning_rate_small
            if np.linalg.norm(x - pre_x) < 1e-6:
                break
        end = time.time()
        report = open(self.output_num, "a+")
        print("total time", end - start, file=report)
        plt.plot(range(len(self.obj_list)), self.obj_list)
        plt.title(learning_rate)
        # plt.show()
        plt.savefig(self.output_fig.split(".png")[0] + "lr" + str(self.learning_rate_large) + ".png")
        # self.cur_obj = np.average(np.array(self.obj_list[-10:]))
        self.cur_obj = self.obj_list[-1]
        print(self.learning_rate_large, self.cur_obj, file=report)
        # print(repr(self.c_uncertain), file=report)
        report.close()
        print(self.cur_obj)
        self.u = x.copy()
        # cur_g = self._compute_gradient(x)
        return scipy.optimize.OptimizeResult(x=x, fun=self.cur_obj, jac=g, nit=ite + 1, nfev=ite + 1,
                                             success=True)

    def optimize_cvar(self, max_iter=2000, lr=0.05):
        self._initialize_control()
        start = time.time()
        self.cur_obj = np.infty
        times = 0
        # self._minimize_u()
        while self.cur_obj > 0.1 and times < 3:
            self._adam(self.init_amps.reshape(-1), max_iter, lr)
            times += 1
        end = time.time()

        if self.pool:
            self.pool.close()
            self.pool.join()

        if len(self.u) != self.n_ts:
            self.u = self.u.reshape((self.n_ts, self.n_ctrls))
        # output the results
        # evo_full_final = self.evolution(self.u)
        penalty = 0
        if self.sum_cons_1:
            penalty = self._compute_l2_penalty(self.u)
        report = open(self.output_num, "a+")
        print("***************** Summary *****************", file=report)
        print("learning rate {} and {} converted at iteration {}".format(self.learning_rate_large,
                                                                         self.learning_rate_small, self.convert_iter),
              file=report)
        print("CVaR parameter and weight parameter", self.thre_cvar, self.weight, file=report)
        print("Final original objective value {}".format(self.obj), file=report)
        print("Final average original objective value {}".format(self.ave_obj), file=report)
        print("Final CVaR objective value {}".format(self.cvar_obj), file=report)
        print("Zeta value {}".format(self.zeta), file=report)
        print("Final maximum original objective value {}".format(max(abs(self.obj))), file=report)
        print("Final original objective value exceeding current objective value {}".format(
            self.obj[self.obj > self.zeta]), file=report)
        print("Final squared penalty error {}".format(penalty), file=report)
        print("Final objective value with all penalty {}".format(self.cur_obj), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=end - start)), file=report)
        print("Computational time {}".format(end - start), file=report)

        # output the control
        if self.obj_mode in ["energy", "energyratio"]:
            self.n_ctrls += 1
            final_amps = np.zeros((self.n_ts, self.n_ctrls))
            final_amps[:, 0] = self.u.copy()
            final_amps[:, 1] = 1 - self.u.copy()

            initial_amps = np.zeros((self.n_ts, self.n_ctrls))
            initial_amps[:, 0] = self.init_amps.copy().reshape(-1)
            initial_amps[:, 1] = 1 - initial_amps[:, 0]

        else:
            initial_amps = self.init_amps.copy()
            final_amps = self.u.copy()

        if self.output_control:
            np.savetxt(self.output_control, final_amps, delimiter=",")

        # output the figures
        time_list = np.array([t * self.delta_t for t in range(self.n_ts + 1)])
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.set_title("Initial control amps")
        # ax1.set_xlabel("Time")
        ax1.set_ylabel("Control amplitude")
        for j in range(self.n_ctrls):
            ax1.step(time_list, np.hstack((initial_amps[:, j], initial_amps[-1, j])), where='post')

        ax2 = fig1.add_subplot(2, 1, 2)
        ax2.set_title("Optimised Control Sequences")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Control amplitude")
        for j in range(final_amps.shape[1]):
            ax2.step(time_list, np.hstack((final_amps[:, j], final_amps[-1, j])), where='post')
        plt.tight_layout()
        if self.output_fig:
            plt.savefig(self.output_fig)
