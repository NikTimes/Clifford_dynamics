import numpy as np 
import random 


"""
One qubit is a superposition of these two states ∣ψ⟩=a∣0⟩+b∣1⟩
where a and b are complex numbers for n qubits, the state is 2^n
To represent the state of a qubit we can store it's coefficients in an array
"""

def initialize_qubits(n):
    # State vector with 2^n components, all initialized to 0, except |00..0⟩
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    return state

GATES = {
    'X': np.array([[0, 1], [1, 0]], dtype=complex),  # Pauli-X (NOT)
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),  # Pauli-Y
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),  # Pauli-Z
    'S': np.array([[1, 0], [0, 1j]], dtype=complex),  # Phase (S) Gate
    'H': (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),  # Hadamard
    'I': np.eye(2, dtype=complex)  # Identity matrix
}

def apply_gate(gate, qubit_idx, state):
    "Apply single qubit gate optimized"
    
    if isinstance(gate, str):
        gate = GATES.get(gate)
        if gate is None:
            raise ValueError(f"Unknown gate: {gate}")

    n_qubits = int(np.log2(len(state)))
    gate_state = state.copy()

    for i in range(len(state)):
        binary_i = format(i, f'0{n_qubits}b')

        if binary_i[qubit_idx] == '0':
            p_index = i + (1 << (n_qubits - qubit_idx - 1)) 
        else:
            p_index = i - (1 << (n_qubits - qubit_idx - 1)) 

        state_i = state[i]
        p_state = state[p_index]

        gate_state[i] = state_i * gate[0, 0] + p_state * gate[0, 1]
        gate_state[p_index] = state_i * gate[1, 0] + p_state * gate[1, 1]
    
    return gate_state


def apply_cnot_gate(control, target, state):
    "Apply CNOT gate to state vector"
    n_qubits = int(np.log2(len(state)))
    cnot_state = state.copy()

    for i in range(n_qubits):
        binary_i = format(i, f'0{n_qubits}b')
        if binary_i[control] == '1':
            
            flipped_state = list(binary_i)
            flipped_state[target] = '1' if flipped_state[target] == '0' else '0'
            new_index = int(''.join(flipped_state), 2)  # Convert back to decimal index

            cnot_state[i], cnot_state[new_index] = state[new_index], state[i]

    return cnot_state


def smeasurement(state):
    n_qubits = int(np.log2(len(state)))
    choice = np.arange(0, len(state), 1)

    probabilities = [abs(x)**2 for x in state]

    collapsed_state = random.choices(choice, weights=probabilities, k=1)[0]

    return format(collapsed_state, f'0{n_qubits}b')