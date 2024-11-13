import numpy as np

"""
Clifford dynamics systems
"""

def g(x1, z1, x2, z2):
    if x1 == 0 and z1 == 0:
        return 0
    elif x1 == 1 and z1 == 1:
        return x2 - z2
    elif x2 == 1 and z2 == 1:
        return z1 * (2 * x1 - 1)
    elif x1 == 1 and z1 == 0 and x2 == 1 and z2 == 0:
        return 0
    elif x1 == 0 and z1 == 1 and x2 == 0 and z2 == 1:
        return 2
    return 0

def rowsum(tableau, h, j):
    n = tableau.shape[0] // 2

    if tableau.dtype != np.int32:
        tableau = tableau.astype(np.int32)

    tableau[h, :2*n] ^= tableau[j, :2*n]  # XOR operation for updating X and Z

    # Update the phase r_h
    new_r_h = 0
    for k in range(n):
        x_hk, z_hk = tableau[h, k], tableau[h, k+n]
        x_jk, z_jk = tableau[j, k], tableau[j, k+n]
        increment = g(x_hk, z_hk, x_jk, z_jk)
        new_r_h += increment
    
    # Modulo 4 for the phase to ensure it stays within valid bounds (0, 1, 2, 3)
    tableau[h, -1] = (tableau[h, -1] + new_r_h) % 4

    return tableau

def initialize_tableau(n):
    # Create a 2n x (2n + 1) tableau with all zeros
    tableau = np.zeros((2 * n, 2 * n + 1), dtype=np.int32)
    
    for i in range(n):
        tableau[i, i] = 1      
        tableau[n + i, n + i] = 1  
    return tableau

def CNOT(tableau, a, b):
    n = len(tableau[0]) // 2
    for i in range(2 * n):
        tableau[i, -1] ^= (tableau[i, a] and tableau[i, n + b]) ^ (tableau[i, n + a] and tableau[i, b])
        tableau[i, b] ^= tableau[i, a]
        tableau[i, n + a] ^= tableau[i, n + b]
    return tableau

def H(tableau, a):
    n = len(tableau[0]) // 2  

    for i in range(2 * n):  
        x, z = tableau[i, a], tableau[i, n + a]  
        tableau[i, -1] ^= (x and z)  # Update phase bit if both X and Z are 1 (Pauli Y)
        tableau[i, a], tableau[i, n + a] = z, x  # Swap X and Z parts for qubit a

    return tableau

def S(tableau, a):
    n = len(tableau[0]) // 2  

    for i in range(2*n):
        tableau[i, -1] ^= (tableau[i, a] and tableau[i, n+a])
        tableau[i, n+a] ^= tableau[i, a]

    return tableau

def measure_random(tableau, a, n, p):
    for j in range(2*n):
        if j != p and tableau[j, a] == 1:
            tableau = rowsum(tableau, j, p)
    tableau[p-n, :] = tableau[p, :]  # Copying p-th row to (p-n)-th row
    tableau[p, :n] = 0  # Reset x values
    tableau[p, n:] = 0  # Reset z values
    tableau[p, a+n] = 1  # Set z_pa
    tableau[p, -1] = np.random.choice([0, 1])  # Random measurement outcome
    return tableau[p, -1]

def measure_deterministic(tableau, a, n):
    new_row = np.zeros((1, tableau.shape[1]))
    tableau = np.vstack([tableau, new_row])
    for j in range(n):
        if tableau[j, a] == 1:
            tableau = rowsum(tableau, 2*n, j + n)  # 2n + 1 is the new row, j + n for the destabilizers
    measurement_outcome = tableau[-1, -1] % 2  # Use modulo to ensure binary outcome
    return measurement_outcome

def measure_qubit(tableau, a):
    n = len(tableau[0]) // 2
    indices = np.where(tableau[n:2*n, a] == 1)[0]

    if indices.size > 0:
        p = np.min(indices) + n  # Correcting the index to the full tableau's context
        return measure_random(tableau, a, n, p)
    else:
        return measure_deterministic(tableau, a, n)

def cmeasurement(tableau):
    n = len(tableau[0]) // 2
    state = []

    for i in range(n):
        state.append(str(measure_qubit(tableau, i)))
    
    return ''.join(state)