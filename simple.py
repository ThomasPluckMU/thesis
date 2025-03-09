import numpy as np

class SIMPLE:
    """
    SIMPLE (Simplified Ising Model with Phase Linked Execution) algorithm
    
    This algorithm combines concepts from both DIMPLE3 and SIMPLE_Grid to create
    a phase-based implementation of an Ising model solver.
    """
    
    def __init__(self, J, h, num_spins=3, buffer_len=32, fsm_threshold=4):
        """
        Initialize the SIMPLE model.
        
        Parameters:
        -----------
        J : numpy.ndarray
            Coupling matrix (doesn't have to be symmetric)
        h : numpy.ndarray
            Local field vector
        num_spins : int
            Number of spins in the system
        buffer_len : int
            Default travel time (baseline) for wavefronts
        fsm_threshold : int
            Threshold for finite state machine transitions
        """
        self.num_spins = num_spins
        self.buffer_len = buffer_len
        self.fsm_threshold = fsm_threshold
        
        # Check if J and h have the correct sizes
        if J.shape != (num_spins, num_spins) or h.shape != (num_spins,):
            raise ValueError("J and h must have correct dimensions")
            
        self.J = J
        self.h = h
        
        # Create wavefront parameters based on coupling strengths
        # Scale J values to determine buffer lengths
        self.off_weights = np.ones((num_spins, num_spins)) * buffer_len
        self.on_weights = np.ones((num_spins, num_spins)) * buffer_len
        
        # Adjust weights based on coupling strengths
        # Positive J -> shorter on_weight, negative J -> longer on_weight
        for i in range(num_spins):
            for j in range(num_spins):
                if J[i, j] != 0:
                    # Scale the weights based on coupling strength
                    # Use a sigmoid-like function to bound the range
                    scaling_factor = 2 / (1 + np.exp(-0.5 * J[i, j])) 
                    self.on_weights[i, j] = int(buffer_len * scaling_factor)
        
        # Initialize state variables
        self.polarity = np.ones(num_spins)  # Maps to the FSM state in SIMPLE_Grid
        self.fsm_states = np.zeros(num_spins)
        
        # Initialize wavefronts (one per spin)
        self.wavefront_positions = np.zeros(num_spins, dtype=int)  # Current cell index
        self.wavefront_local_positions = np.random.randint(0, buffer_len, size=num_spins)  # Position within cell
        
        # Calculate mean period for phase calculations (from SIMPLE_Grid)
        self.mean_weights = (self.on_weights + self.off_weights) / 2
        self.mean_period = np.sum(self.mean_weights, axis=0)
        
    def iterate(self, num_iterations=1):
        """
        Run the SIMPLE iteration process for a specified number of steps.
        
        Parameters:
        -----------
        num_iterations : int
            Number of iterations to run
        """
        for _ in range(num_iterations):
            # Find smallest local position to advance system uniformly
            min_local_pos = np.min(self.wavefront_local_positions)
            
            # Advance all wavefronts to this minimum position
            advances = min_local_pos * np.ones(self.num_spins)
            self.wavefront_local_positions -= advances
            
            # Process wavefronts that have reached the end of their cells
            for i in range(self.num_spins):
                if self.wavefront_local_positions[i] == 0:
                    # Move to next cell
                    next_cell = (self.wavefront_positions[i] + 1) % self.num_spins
                    self.wavefront_positions[i] = next_cell
                    
                    # Determine coupling partner
                    coupled_idx = next_cell
                    
                    # Update FSM states (similar to SIMPLE_Grid)
                    self.fsm_states[i] += 1
                    self.fsm_states[coupled_idx] += 1
                    
                    # Check for state transitions
                    if self.fsm_states[i] >= self.fsm_threshold:
                        self.fsm_states[i] = 0
                        self.polarity[i] *= -1
                        
                    if self.fsm_states[coupled_idx] >= self.fsm_threshold:
                        self.fsm_states[coupled_idx] = 0
                        self.polarity[coupled_idx] *= -1
                    
                    # Determine new cell transit time based on state and coupling
                    curr_weight = self.on_weights[i, next_cell] if self.polarity[i] > 0 else self.off_weights[i, next_cell]
                    
                    # Add field bias from h
                    curr_weight = max(1, int(curr_weight + self.h[i]))
                    
                    # Set new local position
                    self.wavefront_local_positions[i] = curr_weight
                else:
                    # Just add a step to local position to represent time advancement
                    self.wavefront_local_positions[i] += 1
    
    def compute_phases(self):
        """
        Compute the phase of each wavefront (borrowed from SIMPLE_Grid)
        
        Returns:
        --------
        numpy.ndarray: Phase values for each spin
        """
        phases = np.zeros(self.num_spins)
        for i in range(self.num_spins):
            # Sum weights up to current position
            accumulated_weight = np.sum(self.mean_weights[i, 0:self.wavefront_positions[i]])
            # Add fractional position within current cell
            local_fraction = self.wavefront_local_positions[i] / self.mean_weights[i, self.wavefront_positions[i]]
            # Normalize by total period
            phases[i] = (accumulated_weight + local_fraction) / self.mean_period[i]
        return phases
    
    def evaluate(self):
        """
        Evaluate the current Hamiltonian energy of the system.
        
        Returns:
        --------
        float: The Hamiltonian energy value
        """
        # DIMPLE3-style energy calculation
        ising_energy = -0.25 * self.polarity.T @ self.J @ self.polarity - 0.5 * self.h @ self.polarity
        
        # Additional phase-based energy component from SIMPLE_Grid
        phases = self.compute_phases()
        phase_energy = 0
        for i in range(self.num_spins):
            for j in range(self.num_spins):
                if i != j and self.J[i, j] != 0:
                    # Phase difference energy term
                    phase_diff = np.abs(0.5 - np.abs(phases[i] - phases[j]))
                    phase_energy += self.J[i, j] * phase_diff
        
        # Combine both energy terms
        total_energy = ising_energy + 0.1 * phase_energy
        
        return total_energy
    
    def get_state(self):
        """
        Get the current state of the system.
        
        Returns:
        --------
        dict: Current state information
        """
        return {
            'polarity': self.polarity,
            'positions': self.wavefront_positions,
            'local_positions': self.wavefront_local_positions,
            'fsm_states': self.fsm_states,
            'phases': self.compute_phases(),
            'energy': self.evaluate()
        }


# Example usage
if __name__ == "__main__":
    # Number of spins in the system
    num_spins = 3
    
    # Default buffer length
    buffer_len = 32
    
    # FSM threshold
    fsm_threshold = 4
    
    # Coupling matrix
    J = np.array([
        [0, -1, 2],
        [-1, 0, 2],
        [2, 2, 0]
    ])
    
    # Local field vector
    h = np.array([1, 1, -2])
    
    # Create the SIMPLE instance
    model = SIMPLE(J, h, num_spins, buffer_len, fsm_threshold)
    
    # Run iterations
    model.iterate(num_iterations=1000)
    
    # Get and print final state
    result = model.get_state()
    print("Final state:")
    print(f"Polarity: {result['polarity']}")
    print(f"Positions: {result['positions']}")
    print(f"Local positions: {result['local_positions']}")
    print(f"FSM states: {result['fsm_states']}")
    print(f"Phases: {result['phases']}")
    print(f"Energy: {result['energy']}