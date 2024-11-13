from matplotlib import pyplot as plt
import numpy as np 
import random 
from tqdm import tqdm
import time
from collections import Counter
import tracemalloc

from classic import *
from clifford import * 

def randCircuit(n, depth):
    # Initialize the circuit as a list of lists to store string values
    circuit = []

    # Define possible single-qubit gates
    single_qubit_gates = ['H', 'S', 'I']

    for i in range(n):
        row = []
        for j in range(depth):
            if random.random() < 0.7:  # 70% chance to pick a single-qubit gate
                gate = random.choice(single_qubit_gates)
            else:  # 30% chance for a CNOT gate
                target = i
                control = random.choice([q for q in range(n) if q != i])  # Ensure control != target
                gate = [target, control]
            row.append(gate)
        circuit.append(row)
    
   
    return np.array(circuit, dtype=object)


def clifford_circuit(randcircuit):
    n = len(randcircuit)
    depth = len(randcircuit[0])
    tableau = initialize_tableau(n)

    start_time = time.time()  # Start the timer

    for i in range(n):  # Loop over qubits
        for j in range(depth):  # Loop over the depth of the circuit
            gate = randcircuit[i][j]

            # Apply correct gate 
            if gate == 'H':
                tableau = H(tableau, i)  
            elif gate == 'S':
                tableau = S(tableau, i) 
            elif gate == 'I':
                pass  # Identity gate; no operation
            elif isinstance(gate, list) and len(gate) == 2:  
                control = gate[0]
                target = gate[1]
                tableau = CNOT(tableau, control, target)  

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    return cmeasurement(tableau), elapsed_time

def classic_circuit(randcircuit):

    n = len(randcircuit)
    depth = len(randcircuit[0])
    state = initialize_qubits(n)  # Initialize the quantum state for `n` qubits

    start_time = time.time()  # Start the timer
    
    for j in range(depth):  # Loop over the depth of the circuit
        for i in range(n):  # Loop over qubits
            gate = randcircuit[i][j]  
            
            # Apply correct gate 
            if gate == 'H':
                state = apply_gate('H', i, state)  
            elif gate == 'S':
                state = apply_gate('S', i, state)  
            elif gate == 'I':
                pass  # Identity gate; no operation
            elif isinstance(gate, list) and len(gate) == 2:  # Check if it's a CNOT gate
                control = gate[0]
                target = gate[1]
                state = apply_cnot_gate(control, target, state)  

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    return smeasurement(state), elapsed_time

def measure_memory_usage(func, *args):
    """
    Measure the memory usage of a function.
    """
    tracemalloc.start()  # Start tracking memory allocations
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()  # Get current and peak memory usage
    tracemalloc.stop()  
    return result, peak  # Return the result and peak memory usage



num_experiments = 1000
randcircuit =  randCircuit(5, 2) # [[' '],['H'],[' ']]

clifford_results = []
classic_results = []
clifford_times = []
classic_times = []
clifford_memory = []
classic_memory = []

for _ in tqdm(range(num_experiments), desc="Running experiments"):
    c_result, c_mem = measure_memory_usage(clifford_circuit, randcircuit)
    classic_result, classic_mem = measure_memory_usage(classic_circuit, randcircuit)
    
    clifford_results.append(c_result[0])
    clifford_times.append(c_result[1])
    clifford_memory.append(c_mem)

    classic_results.append(classic_result[0])
    classic_times.append(classic_result[1])
    classic_memory.append(classic_mem)

# Convert the counts to a list of dictionaries
clifford_counts = Counter(clifford_results)
clifford_data = [{"Binary String": k, "Count": v} for k, v in clifford_counts.items()]

classic_counts = Counter(classic_results)
classic_data = [{"Binary String": k, "Count": v} for k, v in classic_counts.items()]

# Extract data
clifford_binary_strings = [d['Binary String'] for d in clifford_data]
clifford_binary_string_counts = [d['Count'] for d in clifford_data]
total_clifford_count = sum(clifford_binary_string_counts)
clifford_amplitudes = [count / total_clifford_count for count in clifford_binary_string_counts]

classic_binary_strings = [d['Binary String'] for d in classic_data]
classic_binary_string_counts = [d['Count'] for d in classic_data]
total_classic_count = sum(classic_binary_string_counts)
classic_amplitudes = [count / total_classic_count for count in classic_binary_string_counts]


# Create the histogram
plt.figure(figsize=(10, 6))

bar_width = 0.35  # 
index = range(len(clifford_binary_strings))

# Plotting
plt.bar(index, clifford_amplitudes, width=bar_width, label='Clifford', alpha=0.8)
plt.bar([i + bar_width for i in index], classic_amplitudes, width=bar_width, label='Classic', alpha=0.8)
plt.xlabel('Qubit Measurements')
plt.ylabel('Probability Amplitudes')
plt.title('Comparison of Binary String Occurrences: Clifford vs. Classic')
plt.xticks([i + bar_width / 2 for i in index], clifford_binary_strings)
plt.legend()
plt.show()

clifford_memory_kb = [mem / 1024 for mem in clifford_memory]
classic_memory_kb = [mem / 1024 for mem in classic_memory]

num_experiments = list(range(1, len(clifford_times) + 1))

cumulative_clifford_runtime = np.cumsum(clifford_times)
cumulative_classic_runtime = np.cumsum(classic_times)

cumulative_clifford_memory = np.cumsum(clifford_memory_kb)
cumulative_classic_memory = np.cumsum(classic_memory_kb)

# Plot cumulative runtime comparison
plt.figure(figsize=(12, 6))
plt.plot(num_experiments, cumulative_clifford_runtime, label='Cumulative Clifford Runtime', alpha=0.8)
plt.plot(num_experiments, cumulative_classic_runtime, label='Cumulative Classic Runtime', alpha=0.8)
plt.xlabel('Experiment Number')
plt.ylabel('Cumulative Runtime (seconds)')
plt.title('Cumulative Runtime Comparison: Clifford vs. Classic')
plt.legend()
plt.show()

# Plot cumulative memory usage comparison
plt.figure(figsize=(12, 6))
plt.plot(num_experiments, cumulative_clifford_memory, label='Cumulative Clifford Memory (KB)', alpha=0.8)
plt.plot(num_experiments, cumulative_classic_memory, label='Cumulative Classic Memory (KB)', alpha=0.8)
plt.xlabel('Experiment Number')
plt.ylabel('Cumulative Memory Usage (KB)')
plt.title('Cumulative Memory Usage Comparison: Clifford vs. Classic')
plt.legend()
plt.show()