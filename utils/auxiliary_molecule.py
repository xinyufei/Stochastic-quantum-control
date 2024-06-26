"""
Hamiltonians for different physical qubit systems.
Right now only implements Hamiltonian for SchusterLab transmon qubit.
"""

import numpy as np
from tools.circuitutil import *
from tools.uccsdcircuit import *

# All of these frequencies below are in GHz:
G_MAXA = 2 * np.pi * 0.05
CHARGE_DRIVE_MAXA = 2 * np.pi * 0.1
FLUX_DRIVE_MAXA = 2 * np.pi * 1.5


def get_H0(N, d):
    """Returns the drift Hamiltonian, H0."""
    return np.zeros((d ** N, d ** N))


def _validate_connectivity(N, connected_qubit_pairs):
    """Each edge should be included only once."""
    for (j, k) in connected_qubit_pairs:
        assert 0 <= j < N
        assert 0 <= k < N
        assert j != k
        assert connected_qubit_pairs.count((j, k)) == 1
        assert connected_qubit_pairs.count((k, j)) == 0
        

def get_Hops_and_Hnames(N, d, connected_qubit_pairs):
    """Returns the control Hamiltonian matrices and their labels."""
    hamiltonians, names = [], []
    for j in range(N):
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s charge drive" % j)

        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) @ get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s flux drive" % j)

    _validate_connectivity(N, connected_qubit_pairs)
    for (j, k) in connected_qubit_pairs:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d)
        matrices[k] = get_adagger(d) + get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s-%s coupling" % (j, k))

    return hamiltonians, names


def get_maxA(N, d, connected_qubit_pairs):
    """Returns the maximium amplitudes of the control pulses corresponding to Hops/Hnames."""
    maxA = []
    for j in range(N):
        maxA.append(CHARGE_DRIVE_MAXA)  # max amp for charge drive on jth qubit
        maxA.append(FLUX_DRIVE_MAXA)  # max amp for flux drive on jth qubit

    for (j, k) in connected_qubit_pairs:
        maxA.append(G_MAXA)  # max amp for coupling between qubits j and k

    return maxA


def get_a(d):
    """Returns the matrix for the annihilation operator (a^{\dagger}), truncated to d-levels."""
    values = np.sqrt(np.arange(1, d))
    return np.diag(values, 1)

                   
def get_adagger(d):
    """Returns the matrix for the creation operator (a^{\dagger}), truncated to d-levels."""
    return get_a(d).T  # real matrix, so transpose is same as the dagger


def get_number_operator(d):
    """Returns the matrix for the number operator, a^\dagger * a, truncated to d-levels"""
    return get_adagger(d) @ get_a(d)


def krons(matrices):
    """Returns the Kronecker product of the given matrices."""
    result = [1]
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def get_full_states_concerned_list(N, d):
    states_concerned_list = []
    for i in range(2 ** N):
        bits = "{0:b}".format(i)
        states_concerned_list.append(int(bits, d))
    return states_concerned_list


def generate_molecule_func(N, d, molecule, optimize=True, target=None):
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, int(N / 2), directed=False)
    print(connected_qubit_pairs)
    H0 = get_H0(N, d).astype("complex128")
    Hops, Hnames = get_Hops_and_Hnames(N, d, connected_qubit_pairs)
    states_concerned_list = get_full_states_concerned_list(N, d)
    maxA = get_maxA(N, d, connected_qubit_pairs)

    # for h in Hops:
    #     print((h == h.conj().T).all())
    # exit()

    if not target:
        circuit = get_uccsd_circuit(molecule, optimize=optimize)
        U = get_unitary(circuit).astype("complex128")
        np.savetxt("../result/target/" + molecule + "_target.csv", U)
    else:
        U = np.loadtxt(target, dtype=np.complex_)

    # print(circuit)
    # print(H0.size)
    # print(len(Hops))
    # print(U.size)
    # print(connected_qubit_pairs)
    # for i in range(len(Hops)):
    #     np.savetxt("../hamiltonians/" + molecule + "_controller_" + str(i + 1) + ".csv", Hops[i])
    Hops_new = [Hops[idx].astype("complex128") * maxA[idx] for idx in range(len(Hops))]
    U0 = np.identity(2 ** N).astype("complex128")
    return Hops_new, H0, U0, U


if __name__ == '__main__':
    Hops, H0, U0, U = generate_molecule_func(4, 2, "LiH")
    print(U.conj().T.dot(U))