import qutip.control.pulseoptim as cpo
from qutip.control.optimizer import *

import time
import datetime
import numpy as np
from scipy.linalg import expm, expm_frechet
import scipy.optimize
import matplotlib.pyplot as plt
import qutip.control.pulseoptim as cpo
from qutip import Qobj


class JointCvarOptimizer: # StepCvarOptimization
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

    def build_optimizer(self, d_hamil, c_hamil, init_state, targ_state, n_ts, evo_time, amp_lbound=0, amp_ubound=1,
                        min_grad=1e-6, max_iter_step=500, obj_mode="fid", init_type="ZERO", seed=None, constant=0,
                        initial_control=None, output_num=None, output_fig=None, output_control=None,
                        sos1=False, penalty=10, d_uncertainty=None, c_uncertainty=None, num_scenario=1,
                        thre_cvar=0, prob=[1]):
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

        self.delta_t = evo_time / n_ts

        if self.obj_mode == 'fid':
            self.d_hamil_qobj = Qobj(d_hamil)
            self.c_hamil_qobj = [Qobj(c_hamil_j) for c_hamil_j in c_hamil]
            self.init_state_qobj = Qobj(init_state)
            self.targ_state_qobj = Qobj(targ_state)
            self.n_ctrls = len(self.c_hamil)
            for l in range(self.num_scenario):
                r_d_hamil = (1 + self.d_uncertain[l]) * self.d_hamil_qobj
                r_c_hamil = [(1 + self.c_uncertain[j, l]) * self.c_hamil_qobj[j] for j in range(self.n_ctrls)]
                self.optim = cpo.create_pulse_optimizer(r_d_hamil, r_c_hamil, self.init_state_qobj,
                                                        self.targ_state_qobj, self.n_ts, self.evo_time,
                                                        amp_lbound=self.amp_lbound, amp_ubound=self.amp_ubound,
                                                        dyn_type='UNIT', phase_option="PSU",
                                                        init_pulse_params={"offset": 0}, gen_stats=True)
            self.u = np.zeros((self.n_ts, self.n_ctrls))
        if self.obj_mode == 'energy':
            self.n_ctrls = len(self.c_hamil) - 1
            self._into = [None] * self.num_scenario
            self._onto = [None] * self.num_scenario
            self.u = np.zeros(self.n_ts)

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

    def _time_evolution_energy(self, control_amps):
        for l in range(self.num_scenario):
            self._into[l] = [self.init_state]
            r_c_hamil = [(1 + self.c_uncertain[0, l]) * self.c_hamil[0], (1 + self.c_uncertain[1, l]) * self.c_hamil[1]]
            for k in range(self.n_ts):
                fwd = expm(-1j * (control_amps[k] * r_c_hamil[0] + (1 - control_amps[k]) * r_c_hamil[1])
                           * self.delta_t).dot(self._into[l][k])
                self._into[l].append(fwd)

    def _back_propagation_energy(self, control_amps):
        for l in range(self.num_scenario):
            r_c_hamil = [(1 + self.c_uncertain[0, l]) * self.c_hamil[0], (1 + self.c_uncertain[1, l]) * self.c_hamil[1]]
            self._onto[l] = [self._into[l][-1].conj().T.dot(r_c_hamil[1].conj().T)]
            for k in range(self.n_ts):
                bwd = self._onto[l][k].dot(expm(-1j * (control_amps[self.n_ts - k - 1] * r_c_hamil[0] + (
                        1 - control_amps[self.n_ts - k - 1]) * r_c_hamil[1]) * self.delta_t))
                self._onto[l].append(bwd)

    def _compute_origin_obj(self, control_amps):
        self.obj = np.zeros(self.num_scenario)
        if self.obj_mode == 'fid':
            for l in range(self.num_scenario):
                self.obj[l] = self.optim[l].fid_err_func_compute(control_amps.reshape[self.n_ts, self.n_ctrls])
        if self.obj_mode == 'energy':
            if not (self.u == control_amps).all():
                self._time_evolution_energy(control_amps)
                self._back_propagation_energy(control_amps)
                self.u = control_amps
            for l in range(self.num_scenario):
                self.obj[l] = np.real(self._into[l][-1].conj().T.dot((1 + self.c_uncertain[1, l])
                                                                     * self.c_hamil[1].dot(self._into[l][-1])))

    def _compute_l2_penalty(self, control_amps):
        penalty = sum(np.power(sum(control_amps[t, j] for j in range(self.n_ctrls)) - 1, 2) for t in range(self.n_ts))
        return penalty

    def _compute_obj(self, *args):
        """
        :param args: control list
        :return: error
        """
        control_amps = args[0].copy()[:-1]
        zeta = args[0][-1]
        # control_amps = control_amps.reshape([self.n_ts, self.n_ctrls])
        self._compute_origin_obj(control_amps)
        obj = zeta + 1 / self.thre_cvar * sum(self.prob[l] * max(0.0, self.obj[l] - zeta)
                                              for l in range(self.num_scenario))
        penalized = 0
        if self.sum_cons_1:
            penalized = self.penalty * self._compute_l2_penalty(control_amps.reshape([self.n_ts, self.n_ctrls]))
        return obj + penalized

    def _compute_gradient(self, *args):
        control_amps = args[0].copy()[:-1]
        zeta = args[0][-1]

        grad_control = np.zeros((self.n_ts, self.n_ctrls))
        penalized_grad = np.zeros((self.n_ts, self.n_ctrls))
        if self.obj_mode == 'fid':
            for l in range(self.num_scenario):
                if zeta <= self.obj[l]:
                    grad = self.optim[l].fid_err_grad_compute(control_amps.reshape([self.n_ts, self.n_ctrls]))
                    grad_control += self.prob[l] / self.thre_cvar * grad

            if self.sum_cons_1:
                for t in range(self.n_ts):
                    grad_p = 2 * self.penalty * (sum(control_amps[t, j] for j in range(self.n_ctrls)) - 1)
                    for j in range(self.n_ctrls):
                        penalized_grad[t, j] = grad_p

        if self.obj_mode == 'energy':
            for l in range(self.num_scenario):
                if zeta <= self.obj[l]:
                    grad = []
                    r_c_hamil = [(1 + self.c_uncertain[0, l]) * self.c_hamil[0],
                                 (1 + self.c_uncertain[1, l]) * self.c_hamil[1]]
                    for k in range(self.n_ts):
                        grad += [-np.imag(self._onto[l][self.n_ts - k - 1].dot((r_c_hamil[1] - r_c_hamil[0]).dot(
                            self._into[l][k + 1])) * self.delta_t)]
                    grad_control += self.prob[l] / self.thre_cvar * np.expand_dims(np.array(grad), 1)

        # gradient for zeta
        grad_zeta = 1.0
        for l in range(self.num_scenario):
            if zeta <= self.obj[l]:
                grad_zeta -= self.prob[l] / self.thre_cvar

        return np.append(grad_control.flatten() + penalized_grad.flatten(), grad_zeta)

    def _minimize_u(self):
        self.time_optimize_start_step = time.time()
        self.num_iter_step = 0
        min_grad = self.min_grad
        results = scipy.optimize.fmin_l_bfgs_b(self._compute_obj, np.append(self.init_amps.reshape(-1), -1.0),
                                               bounds=[(self.amp_lbound, self.amp_ubound)] * self.n_ts * self.n_ctrls +
                                               [(-np.infty, np.infty)], pgtol=min_grad, fprime=self._compute_gradient,
                                               maxiter=self.max_iter_step)
        if self.n_ctrls == 1:
            self.u = results[0][:-1]
        else:
            self.u = results[0][:-1].reshape((self.n_ts, self.n_ctrls)).copy()
        self.zeta = results[0][-1]
        self.cur_obj = results[1]
        self.cur_grad = results[2]['grad']
        self.result = results
        # self.u = results.x.reshape((self.n_ts, self.n_ctrls)).copy()
        # self.cur_obj = results.fun

    def optimize_cvar(self):
        self._initialize_control()
        start = time.time()
        self._minimize_u()
        end = time.time()

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
        report = open(self.output_num, "w+")
        # print("Final evolution\n{}\n".format(evo_full_final), file=report)
        print("********* Summary *****************", file=report)
        print("Zeta value {}".format(self.zeta), file=report)
        print("Final original objective value {}".format(self.obj), file=report)
        print("Final original objective value exceeding current objective value {}".format(
            self.obj[self.obj > self.cur_obj]), file=report)
        print("Final squared penalty error {}".format(penalty), file=report)
        print("Final penalized objective value {}".format(self.cur_obj), file=report)
        print("Final gradient {}".format(self.result[2]['grad']), file=report)
        print("Number of iterations {}".format(self.result[2]['nit']), file=report)
        print("Terminated due to {}".format(t_reason), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=end - start)), file=report)
        print("Computational time {}".format(end - start), file=report)

        # output the control
        if self.obj_mode == "energy":
            self.n_ctrls += 1
            final_amps = np.zeros((self.n_ts, self.n_ctrls))
            final_amps[:, 0] = self.u.copy()
            final_amps[:, 1] = 1 - self.u.copy()

            initial_amps = np.zeros((self.n_ts, self.n_ctrls))
            initial_amps[:, 0] = self.init_amps.copy().reshape(-1)
            initial_amps[:, 1] = 1 - initial_amps[:, 0]

        else:
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
