import numpy as np

class DIMPLE3:
    def __init__(self, J, h, num_spins=3, buffer_chain_len=10):
        """
        Initialize the DIMPLE3 model.
        
        Parameters:
        -----------
        J : numpy.ndarray
            Coupling matrix (doesn't have to be symmetric)
        h : numpy.ndarray
            Local field vector
        num_spins : int
            Number of spins in the system
        buffer_chain_len : int
            Default travel time (baseline)
        """
        self.num_spins = num_spins
        self.buffer_chain_len = buffer_chain_len
        
        # Check if J and h have the correct sizes
        if J.shape != (num_spins, num_spins) or h.shape != (num_spins,):
            raise ValueError("Check J and h are correct size")
            
        self.J = J
        self.h = h
        
        # Initialize state variables
        self.counters = np.random.randint(0, buffer_chain_len, size=num_spins)
        self.location = np.random.randint(0, num_spins, size=num_spins)
        self.polarity = np.random.choice([-1,+1], size=num_spins)
        
    def iterate(self):
        """
        Run the DIMPLE3 iteration process.
        """
        # Calculate common drift and adjust counters
        common_drift = np.min(self.counters)
        self.counters -= common_drift
        
        # Loop through the counters
        for i, v in enumerate(self.counters): 
            # I've run out of buffer chain, time for a new coupling cell
            if v == 0:
                # Behavior at the end of the buffer chain, ie. at the inverter
                if self.location[i] == self.num_spins - 1:
                    self.polarity[i] *= -1
                    self.location[i] = 0
                    self.counters[i] = self.h[i]
                # Behavior on the buffer chain
                else:
                    self.location[i] += 1
                
                # Get the location index to compare with
                compare_idx = self.location[i]
                
                # Comparison logic: is the coupled wave front ahead or behind me?
                if self.location[compare_idx] > self.location[i]:
                    # If ahead, is it a different polarity? If so, trigger the XOR!
                    if self.polarity[compare_idx] != self.polarity[i]:
                        self.counters[i] = self.buffer_chain_len + self.J[i][compare_idx]
                    else:
                        self.counters[i] = self.buffer_chain_len
                else:
                    # If behind, is it the same polarity? If so, trigger the XOR!
                    if self.polarity[compare_idx] == self.polarity[i]:  # Fixed: != to ==
                        self.counters[i] = self.buffer_chain_len + self.J[i][compare_idx]
                    else:
                        self.counters[i] = self.buffer_chain_len
                        
    def evaluate(self):
        """
        Evaluated the current Hamiltonian energy of the system in relation to the underlying QUBO.

        Returns:
            numpy.array: the Hamiltonian score given by the current state of the system
        """
        return - 0.25 * self.polarity.T @ self.J @ self.polarity - 0.25 * np.sum(self.J)

# Example usage
if __name__ == "__main__":
    # Number of spins in the system
    num_spins = 3
    
    # Default travel time, call it baseline
    buffer_chain_len = 10
    
    # Good ol' QUBO coupling params
    # NOTE: these don't have to be symmetric!
    J = np.array([
        [0, -1, 2],
        [-1, 0, 2],
        [2, 2, 0]
    ])
    
    h = np.array([
        [1],
        [1],
        [-2]
    ])
    
    # Create the DIMPLE3 instance
    model = DIMPLE3(J, h, num_spins, buffer_chain_len)
    
    # Run iterations
    for i in range(1000):
        model.iterate(num_iterations=1000)
        
    result = {
        'counters':model.counters,
        'location':model.location,
        'polarity':model.polarity
        }
    
    # Print final state
    print("Final state:")
    print(f"Counters: {result['counters']}")
    print(f"Location: {result['location']}")
    print(f"Polarity: {result['polarity']}")