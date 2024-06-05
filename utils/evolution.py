import numpy as np
from scipy.linalg import expm
import multiprocessing


def time_evolution(H_d, H_c, n_ts, evo_time, u_list, X_0, sum_cons_1, ops_max_amp):
    if not isinstance(ops_max_amp, list):
        max_amp = [ops_max_amp] * len(H_c)
    else:
        max_amp = ops_max_amp
    H_origin_c = [h_c.copy() for h_c in H_c]
    if sum_cons_1:
        H_d_new = H_d + H_origin_c[-1]
        H_d = H_d_new.copy()
        H_c = [(H_origin_c[i] - H_origin_c[-1]).copy() for i in range(len(H_origin_c) - 1)]

    n_ctrls = len(H_c)
    delta_t = evo_time / n_ts
    X = [X_0]
    for t in range(n_ts):
        H_t = H_d.copy()
        for j in range(n_ctrls):
            H_t += u_list[t, j] * max_amp[j] * H_c[j].copy()
        X_t = expm(-1j * H_t * delta_t).dot(X[t])
        X.append(X_t)
    return X[-1]


def time_evolution_dependent(H_d, H_c, n_ts, evo_time, u_list, X_0):
    n_ctrls = len(H_c)
    delta_t = evo_time / n_ts
    X = [X_0]
    for t in range(n_ts):
        H_t = H_d.copy()
        for j in range(n_ctrls):
            H_t += u_list[t, j] * H_c[j][t].copy()
        X_t = expm(-1j * H_t * delta_t).dot(X[t])
        X.append(X_t)
    return X[-1]


def compute_obj_fid(U_targ, U_result):
    # fid = np.abs(np.trace((np.linalg.inv(U_targ.full()).dot(U_result)))) / U_targ.full().shape[0]
    # fid = np.abs(np.trace(((U_targ.full()).dot(U_result)))) / U_targ.full().shape[0]
    fid = np.abs(np.trace((U_targ.full().conj().T.dot(U_result)))) / U_targ.full().shape[0]
    obj = 1 - fid
    return obj


def compute_obj_energy(C, X_result):
    obj = np.real(np.real(X_result.conj().T.dot(C.dot(X_result))))
    return obj


def compute_obj_with_TV(U_targ, U_result, u_list, n_ctrls, alpha):
    fid = np.abs(np.trace((np.linalg.inv(U_targ.full()).dot(U_result)))) / U_targ.full().shape[0]
    TV = sum(sum(abs(u_list[time_step + 1, j] - u_list[time_step, j]) for time_step in range(u_list.shape[0] - 1))
             for j in range(u_list.shape[1]))
    return 1 - fid + alpha * TV


def compute_TV_norm(u_list):
    TV = sum(sum(abs(u_list[time_step + 1, j] - u_list[time_step, j]) for time_step in range(u_list.shape[0] - 1))
             for j in range(u_list.shape[1]))
    return TV


def compute_sum_cons(u_list, max_controllers):
    n_ctrls = u_list.shape[1]
    n_ts = u_list.shape[0]
    penalty = sum(np.power(sum(u_list[t, j] for j in range(n_ctrls)) - max_controllers, 2) for t in range(n_ts))
    return penalty


def out_of_sample_test(num_scenario, c_uncertainty, H_d, H_c, n_ts, evo_time, u_list, X_0, X_targ, obj_type,
                       true_energy, binary=True):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    obj_val = pool.starmap(out_of_sample_test_single,
                           [(H_d, c_uncertainty[:, :, l], H_c, H_c[1],
                             # [[(1 + c_uncertainty[j, k, l]) * H_c[j] for k in range(n_ts)]
                             #       for j in range(len(H_c))], H_c[1],
                             n_ts, evo_time, u_list, X_0, X_targ, obj_type, true_energy, binary)
                            for l in range(num_scenario)])
    pool.close()
    pool.join()
    return obj_val


def out_of_sample_test_single(H_d, uncertainty, H_c, true_h_c, n_ts, evo_time, u_list, X_0, X_targ, obj_type,
                              true_energy, binary=True):
    if binary and np.linalg.norm(H_d) == 0:
        cur_result = time_evolution_binary(H_d, uncertainty, H_c, n_ts, evo_time, u_list, X_0, 1)
    else:
        cur_result = time_evolution_dependent(H_d, [[(1 + uncertainty[j, k]) * H_c[j] for k in range(n_ts)]
                                                    for j in range(len(H_c))], n_ts, evo_time, u_list, X_0)
    if obj_type == 'energyratio':
        energy = compute_obj_energy(true_h_c, cur_result)
        obj = 1 - energy / true_energy
    if obj_type == 'energy':
        energy = compute_obj_energy(true_h_c, cur_result)
        obj = energy
    if obj_type == 'fid':
        obj = compute_obj_fid(X_targ, cur_result)
    return obj


def obj_of_uncertain(epsilon, H_d, H_c, n_ts, evo_time, u_list, X_0, X_targ, obj_type, true_energy):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    obj = pool.starmap(out_of_sample_test_single, [(H_d, [(1 + epsilon_0) * H_c[0], (1 + epsilon_1) * H_c[1]], H_c[1],
                                                    n_ts, evo_time, u_list, X_0, X_targ, obj_type, true_energy)
                                                   for (epsilon_0, epsilon_1) in zip(*(x.flat for x in epsilon))])
    pool.close()
    pool.join()
    # if isinstance(epsilon, list):
    #     cur_result = time_evolution(H_d, [(1 + s_epsilon) * h_c.full() for (h_c, s_epsilon) in zip(H_c, epsilon)],
    #                                 n_ts, evo_time, u_list, X_0, False, 1)
    # else:
    #     cur_result = time_evolution(H_d, [(1 + epsilon) * h_c.full() for h_c in H_c],
    #                                 n_ts, evo_time, u_list, X_0, False, 1)
    # if obj_type == 'energyratio':
    #     energy = compute_obj_energy(H_c[1], cur_result)
    #     obj = 1 - energy / true_energy
    # if obj_type == 'energy':
    #     energy = compute_obj_energy(H_c[1], cur_result)
    #     obj = energy
    # if obj_type == 'fid':
    #     obj = compute_obj_fid(X_targ, cur_result)
    return obj


def obj_of_uncertain_fid(epsilon, H_d, H_c, n_ts, evo_time, u_list, X_0, X_targ):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    obj = pool.starmap(out_of_sample_test_single, [((1 + epsilons) * H_d, [(1 + epsilons) * hc for hc in H_c], None,
                                                    n_ts, evo_time, u_list, X_0, X_targ, 'fid', None)
                                                   for epsilons in epsilon])
    pool.close()
    # pool.join()
    return obj


def time_evolution_binary(H_d, uncertainty, H_c, n_ts, evo_time, u_list, X_0, ops_max_amp):
    if not isinstance(ops_max_amp, list):
        max_amp = [ops_max_amp] * len(H_c)
    else:
        max_amp = ops_max_amp

    n_ctrls = len(H_c)
    delta_t = evo_time / n_ts
    exp_q = []
    exp_diag = []
    # expH = []
    for (j, hc) in enumerate(H_c):
        # expH.append(expm(-1j * (hc * max_amp[j] + H_d) * delta_t))
        s, v = np.linalg.eigh(hc * max_amp[j] + H_d)
        exp_q.append(v)
        exp_diag.append(s)

    X = [X_0]
    for t in range(n_ts):
        active_ctrl = int(np.argwhere(u_list[t, :] == 1).flatten()[0])
        v, s = exp_q[active_ctrl], exp_diag[active_ctrl]
        # X_t = sum(expH[int(j)] for j in active_ctrl).dot(X[t])
        X_t = v.dot(np.diag(np.exp(-1j * s * delta_t * (1 + uncertainty[active_ctrl, t])))).dot(v.conj().T).dot(X[t])
        X.append(X_t)
    return X[-1]
