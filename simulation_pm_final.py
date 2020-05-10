import numpy as np
import sys
import os
from multiprocessing import Process
from scipy.stats import entropy

# command line arguments

n_matrices = int(sys.argv[1])
n_samples = int(sys.argv[2])
mnps = int(sys.argv[3])

# directories

omega_dir = 'eigenfrequencies'

# set seed
seed = 42
np.random.seed(seed)

# other arguments
# n_samples = 50
d = 5
ks = [2, 3, 4, 5]
omegas = np.random.rand(5)
# this takes the value of the smallest difference in frequences
# corresponding to the largest period
min_freq = float('inf')
for i in range(len(omegas)):
    for j in range(len(omegas)):
        if i != j:
            val1 = omegas[i]
            val2 = omegas[j]
            if abs(val1 - val2) < min_freq:
                min_freq = abs(val1 - val2)
omegas *= 2 * np.pi # convert to an angular frequency
omegas /= min_freq # ensure the largest period is T = 1

try:
    cwd = os.getcwd()
    new_dir = os.path.join(cwd, omega_dir)
    os.mkdir(new_dir)

except FileExistsError:
    pass

np.savetxt(f'{omega_dir}/seed={seed}.csv', omegas)
omegas = list(omegas)
ts = np.linspace(0, 1, n_samples)


class DataSampler:

    '''
    A class that takes a fixed number of random samples of an interference
    pattern for a given d-dimensional Hilbert space, coherence rank k and a
    fixed number of density matrices.
    '''

    def __init__(self, d, k, density_matrices, entropies_k, n_samples, omegas=False, ts=False):

        self.d = d # dimensions of Hilbert space
        self.k = k # coherence rank
        self.d_mat_list = density_matrices # list of density matrices
        self.entropies = entropies_k
        self.n_samples = n_samples # number of samples to take
        self.omegas = omegas
        self.ts = ts
        # list of interference pattern amplitudes
        self.Ps = np.zeros((n_matrices, n_samples), dtype=complex)


    def get_phis(self, omegas=False, ts=False):

        ''' Generate all necessary vectors of random phases. '''

        if not omegas:
            self.phis = [phi(self.d) for _ in range(self.n_samples)]
        else:
            omegas = -np.array(omegas)
            self.phis = [omegas * t for t in self.ts]


    def get_Ps(self):

        '''
        Find the amplitudes of the interference pattern at a discrete set of
        phase vectors.
        '''

        #Cycle through all the density matricies
        for matrix_index, d_mat in enumerate(self.d_mat_list):
            #Cycle through all the phase vectors
            for phase_vector_index, phi_vec in enumerate(self.phis):
                #Cycle through all relative phase vector components twice
                self.Ps[matrix_index, phase_vector_index] = P(d_mat, phi_vec, self.d)


    def sample(self):

        ''' Read in the data and then sample the interference pattern. '''

        self.get_phis(self.omegas)
        self.get_Ps()

        np.savetxt(f'{self.d}_{self.k}.csv', np.real(self.Ps))
        np.savetxt(f'{self.d}_{self.k}_entropies.csv', np.real(self.entropies))


def P(d_mat, phi, d):

    '''
    Find the amplitudes of the interference pattern at different phase vectors.
    '''

    out = 0

    # calculate the P values
    for j in range(d):
        for m in range(d):
            if j != m:
                out += d_mat[j, m] * np.exp(complex(0,1) * (phi[j] - phi[m]))

    return 1 + out


def phi(d):

    ''' Create a vector of randomly generated phases from 0 to 2 pi. '''

    # return  np.concatenate(np.array([0]), 2 * np.pi * np.random.rand(d-1))
    return  np.append(np.array([0]), 2 * np.pi * np.random.rand(d-1))


def state_vector(d, k):

    '''
    Generate a state vector with rank k coherence in d-dimensional Hilbert
    space.
    '''

    # coherence of rank k
    v = np.concatenate((np.random.rand(k), np.zeros(d-k)))

    # normalise
    v /= np.sqrt(v.dot(v))

    # random phase
    phases = np.array(2 * np.pi * np.random.rand(d), dtype=complex)
    v = np.array(v, dtype=complex) * np.array(np.exp(complex(0, 1) * phases))

    # shuffle components
    np.random.shuffle(v)

    return v


def pure_density_matrix(v):

    ''' Generate the density matrix for a pure state. '''

    # space dimensions
    d = len(v)

    # density matrix elements are |state><state|
    return np.array([[v[i] * v[j].conj() for j in range(d)] for i in range(d)], dtype=complex)


def mixed_density_matrix(d, k, max_n_pure_states):


    n_pure_states = np.random.choice(np.arange(1, max_n_pure_states+1))
    # n_pure_states = max_n_pure_states
    # n = max_n_pure_states

    ks = np.arange(2, k+1)
    ranks = np.append(np.array([k]), np.random.choice(ks, n_pure_states-1))

    probabilities = np.random.rand(len(ranks))
    probabilities /= sum(probabilities)

    # Generate ensemble of probabilities
    # probabilities = np.array([0.85,  *np.random.exponential(0.075, n-1)])
    # probabilities = np.random.rand(n)
    # probabilities /= sum(probabilities)
    ent = entropy(probabilities)

    out = np.zeros((d, d), dtype=complex)

    for p, rank in zip(probabilities, ranks):

        v = state_vector(d, rank)
        pure_rho = pure_density_matrix(v)
        out += p * pure_rho

    return out, ent


def execute1(density_matrices, entropies, d, k, n, mixed, max_n_pure_states):

    density_matrices[k] = []
    entropies[k] = []

    # n density matrices for each rank
    print(f'Generating density matrices for k={k}...')
    for _ in range(n):

        if mixed:
            mat, ent = mixed_density_matrix(d, k, max_n_pure_states)
            density_matrices[k].append(mat)
            entropies[k].append(ent)

        else:
            v = state_vector(d, k)
            density_matrices[k].append(pure_density_matrix(v))


def get_density_matrices(n, d, ks, mixed, max_n_pure_states=None):

    ''' Generate n density matrices for pure states of each coherence rank. '''

    # dictionary to store density matrices for each coherence rank
    density_matrices = dict()
    entropies = dict()

    processes = []

    # iterate over all coherence ranks
    for k in ks:
    #     p = Process(target=execute1, args=[density_matrices, d, k, n, mixed, max_n_pure_states])
    #     p.start()
    #     processes.append(p)
    #
    # for process in processes:
    #     process.join()

        execute1(density_matrices, entropies, d, k, n, mixed, max_n_pure_states)

    return dict(sorted(density_matrices.items())), dict(sorted(entropies.items()))



def execute2(samplers, d, k, density_matrices, entropies, n_samples, omegas):
    ds = DataSampler(d, k, density_matrices[k], entropies[k], n_samples, omegas, ts)
    ds.sample()
    # samplers.append(ds)


def ramsey(d, ks, density_matrices, entropies, n_samples, omegas):

    ''' Simulate the Ramsey sequence for each coherence rank. '''

    print('Generating interference values...')

    # add the data samplers for each coherence rank to a blank list
    samplers = []

    processes = []

    for k in ks:

        print(f'k={k}...')

        p = Process(target=execute2, args=[samplers, d, k, density_matrices, entropies, n_samples, omegas])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    #    execute2(samplers, d, k, density_matrices, n_samples, save_dir, omegas)


if __name__ == '__main__':

    mixed = True

    density_matrices, entropies = get_density_matrices(n_matrices, d, ks, mixed, mnps)
    ramsey(d, ks, density_matrices, entropies, n_samples, omegas)
