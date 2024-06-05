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
# print(scipy.__file__)
from scipy.linalg import expm, expm_frechet
import scipy.optimize
import gurobipy as gb
import multiprocessing


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


class DirectCvarOptimizer:
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
                        sos1=False, penalty=10, d_uncertainty=None, c_uncertainty=None, num_scenario=1,
                        thre_cvar=0, prob=[1], weight=0, true_energy=None):
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
        self.d_uncertain = d_uncertainty
        self.c_uncertain = c_uncertainty
        self.num_scenario = num_scenario
        self.thre_cvar = thre_cvar
        self.prob = prob
        self.weight = weight

        self.delta_t = evo_time / n_ts

        self.r_d_hamil = [None] * num_scenario
        self.r_c_hamil = [None] * num_scenario

        if self.obj_mode == 'fid':
            self.d_hamil_qobj = Qobj(d_hamil)
            self.c_hamil_qobj = [Qobj(c_hamil_j) for c_hamil_j in c_hamil]
            self.init_state_qobj = Qobj(init_state)
            self.targ_state_qobj = Qobj(targ_state)
            self.n_ctrls = len(self.c_hamil)

            for l in range(self.num_scenario):
                self.r_d_hamil[l] = [(1 + self.d_uncertain[k, l]) * self.d_hamil_qobj for k in range(self.n_ts)]
                self.r_c_hamil[l] = [[(1 + self.c_uncertain[j, k, l]) * self.c_hamil_qobj[j]
                                      for j in range(self.n_ctrls)] for k in range(self.n_ts)]
                self.optim.append(cpo.create_pulse_optimizer(self.r_d_hamil[l], self.r_c_hamil[l], self.init_state_qobj,
                                                             self.targ_state_qobj, self.n_ts, self.evo_time,
                                                             amp_lbound=self.amp_lbound, amp_ubound=self.amp_ubound,
                                                             dyn_type='UNIT', phase_option="PSU",
                                                             init_pulse_params={"offset": 0}, gen_stats=True))
            self.u = np.zeros((self.n_ts, self.n_ctrls))
            self.pool = multiprocessing.Pool(processes=7)
        if self.obj_mode in ['energy', 'energyratio']:
            self.n_ctrls = len(self.c_hamil) - 1
            self._into = [None] * self.num_scenario
            self._onto = [None] * self.num_scenario
            self.u = np.zeros(self.n_ts)
            self.mean_true_energy = true_energy
            self.pool = multiprocessing.Pool(processes=7)

        self.obj = np.zeros(num_scenario)
        self.choose_scenario = range(self.num_scenario)
        self.sample_size = self.num_scenario

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
        if self.obj_mode == 'fid':
            for l in range(self.num_scenario):
                dyn = self.optim[l].dynamics
                dyn.initialize_controls(self.init_amps)
                print(dyn.time_depend_ctrl_dyn_gen, dyn._num_ctrls)

    def _time_evolution_energy(self, control_amps):
        self._into = self.pool.starmap(time_evolution_energy_each_scenario,
                                       [(self.init_state, self.c_uncertain[:, :, l], self.c_hamil, control_amps,
                                         self.delta_t) for l in self.choose_scenario])

    def _back_propagation_energy(self, control_amps):
        self._onto = self.pool.starmap(back_propagation_energy_each_scenario,
                                       [(self._into[l][-1], self.c_uncertain[:, :, l],
                                         control_amps, self.delta_t, self.c_hamil) for l in self.choose_scenario])

    def _compute_origin_obj(self, control_amps, recompute=False):
        change = np.array_equal(self.u, control_amps)
        if not recompute and change:
            return self.iter_obj
        self.obj = np.zeros(self.num_scenario)
        reshape_ctrl = control_amps.reshape(self.n_ts, self.n_ctrls)
        if self.obj_mode == 'fid':
            self.obj = np.array(self.pool.starmap(compute_fid_obj, [(self.optim[l], reshape_ctrl)
                                                                    for l in self.choose_scenario]))
        if self.obj_mode in ['energy', 'energyratio']:
            if not (self.u == control_amps).all():
                self._time_evolution_energy(control_amps)
                self._back_propagation_energy(control_amps)
                self.u = control_amps
                if self.obj_mode == "energy":
                    self.obj = [np.real(self._into[idx][-1].conj().T.dot(self.c_hamil[1].dot(self._into[idx][-1])))
                                for idx in range(self.sample_size)]
                if self.obj_mode == "energyratio":
                    self.obj = [1 - np.real(self._into[idx][-1].conj().T.dot(self.c_hamil[1].dot(self._into[idx][-1])))
                                / self.mean_true_energy for idx in range(self.sample_size)]
                self.obj = np.array(self.obj)
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
            all_grad = self.pool.starmap(compute_fid_gradient, [(self.optim[l], reshape_control)
                                                                for l in self.choose_scenario])
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
                                           self.c_hamil, self.n_ts, self.delta_t) for l in self.choose_scenario])
            for idx in range(self.sample_size):
                if self.zeta < self.obj[idx]:
                    cvar_grad_control += 1 / self.sample_size / self.thre_cvar * all_grad[idx]
                    exceed_idx.append(idx)
                ave_grad_control += 1 / self.sample_size * all_grad[idx]

        if self.obj_mode == 'energyratio':
            all_grad = self.pool.starmap(
                compute_energy_gradient, [(self._onto[l], self._into[l], self.c_uncertain[:, :, l],
                                           self.c_hamil, self.n_ts, self.delta_t) for l in self.choose_scenario])
            for idx in range(self.sample_size):
                if self.zeta < self.obj[idx]:
                    cvar_grad_control += 1 / self.sample_size / self.thre_cvar * all_grad[idx] / (
                        -self.mean_true_energy)
                    exceed_idx.append(idx)
                ave_grad_control += 1 / self.sample_size * all_grad[idx]

        l_star_coeff = 1 / self.sample_size * len(exceed_idx) / self.thre_cvar
        if l_star_coeff != 1:
            cvar_grad_control += (1 - l_star_coeff) * all_grad[self.l_star]

        grad_control = self.weight * ave_grad_control + (1 - self.weight) * cvar_grad_control

        return grad_control.flatten() + penalized_grad.flatten()

    def _step_call_back(self, *args):
        wall_time_step = time.time() - self.time_optimize_start_step
        report = open(self.output_num, "a+")
        print("Iteration", self.num_iter_step, "completed with objective value", self.iter_obj, file=report)
        report.close()
        self.num_iter_step += 1
        pass

    def _minimize_u(self):
        self.time_optimize_start_step = time.time()
        self.num_iter_step = 0
        min_grad = self.min_grad

        ftol = 1e7
        if self.obj_mode == 'energyratio':
            ftol = 1e10
        results = fmin_l_bfgs_b(self._compute_obj, self.init_amps.reshape(-1),
                                bounds=[(self.amp_lbound, self.amp_ubound)] * self.n_ts * self.n_ctrls,
                                pgtol=min_grad,
                                fprime=self._compute_gradient,
                                # approx_grad=1, epsilon=1e-14,
                                factr=ftol,
                                maxiter=self.max_iter_step, maxls=20,
                                callback=self._step_call_back)
        # , iprint=101)
        if self.n_ctrls == 1:
            self.u = results[0]
        else:
            self.u = results[0].reshape((self.n_ts, self.n_ctrls)).copy()

        self.cur_obj = results[1]
        self.cur_grad = results[2]['grad']
        self.result = results

    def optimize_cvar(self):
        self._initialize_control()
        start = time.time()
        self._minimize_u()
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
        if self.result[2]['warnflag'] == 0:
            t_reason = "Converged"
        if self.result[2]['warnflag'] == 1:
            t_reason = "Maximum iteration"
        if self.result[2]['warnflag'] == 2:
            t_reason = self.result[2]['task']
        report = open(self.output_num, "a+")
        # print("Final evolution\n{}\n".format(evo_full_final), file=report)
        print("***************** Summary *****************", file=report)
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
        print("Final gradient {}".format(self.result[2]['grad']), file=report)
        print("Number of iterations {}".format(self.result[2]['nit']), file=report)
        print("Terminated due to {} with task {}".format(t_reason, self.result[2]['task']), file=report)
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
