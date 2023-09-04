import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix, init_printing, sqrt, pprint, Matrix
from sympy.physics.quantum import TensorProduct

def grover(Q_BITS, R):

    a = np.identity(2 ** Q_BITS, dtype=int)
    a[R-1][R-1] = -1
    ORACLE = Matrix(a)
    # define Hadamard
    H = 1 / sqrt(2) * Matrix([[1, 1], [1, -1]])
    H_n = TensorProduct(*([H] * Q_BITS))
    R_n = Matrix(np.identity(2 ** Q_BITS, dtype=int))
    R_n[0] = -1
    D_n = -H_n * R_n * H_n
    STATE = Matrix([1] + [0] * ((2 ** Q_BITS) - 1))

    STATE = H_n * STATE
    # use the oracle
    STATE = ORACLE * STATE   
    # use the mirror on average
    STATE = D_n * STATE   

    print(STATE, "\n Grover's search result for",
        R,"th input is", STATE[R-1],
        "\n-------------------------------------------")
    # STATE = ORACLE * STATE
    # STATE = D_n * STATE
    return STATE

def bar_plot(N, val):
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_ylabel('Prob')
    ax.set_title('Grovers Search Result')
    ax.grid()
    ax.bar([i for i in range(1, 2**N + 1)], val)
    plt.show()


"""
PART 1
  Run Grover's Algorithm for 4-Qubits circuit

"""
probs_1 = grover(4, 12)
bar_plot(4, probs_1)


"""
PART 2
  Run Grover's Algorithm for 10-Qubits circuit
"""
probs_2 = grover(10, 500)
bar_plot(10, probs_2)