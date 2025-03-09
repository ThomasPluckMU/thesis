import io
import itertools
import os
import numpy as np
import pandas as pd
import networkx as nx

def preprocess_tcarp(
    # File to be processed
    file: io.TextIOWrapper,
    # Problem Definition
    num_vehicles: int = 1,
    time_capacity: int = 1,
    # QUBO Parameters
    mu: float = 1.0,
    lambd: float = -1.0
):
    # Handle raw data
    content = file.read().split("Cost")[-1].split("\n")[1:-1]
    content = [[int(i) for i in c.split()] for c in content]
    content = np.array(content)[:, 1:]
    
    # Roll into Pandas dataframe
    header = ['from', 'to', 'demand', 'cost']
    df = pd.DataFrame(content, columns=header)
    
    # Instantiate graph
    G = nx.DiGraph()
    
    # Find maximum node ID for dual_VRP dimensions
    max_node = max(df['from'].max(), df['to'].max())
    
    # Correctly initialize dual_VRP with proper dimensions
    num_arcs = len(df)
    dual_VRP = np.zeros((num_arcs, num_arcs))
    
    # Add edges with weights from the DataFrame
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'], weight=row['cost'])
    
    # Calculate dual VRP graph
    for i in range(num_arcs):
        for j in range(num_arcs):
            # Construct dual VRP coupling
            source = df.iloc[i]['to']
            target = df.iloc[j]['from']
            
            try:
                # Find the shortest path
                path_length = nx.shortest_path_length(
                    G,
                    source=source,
                    target=target,
                    weight='weight'
                )
            except nx.NetworkXNoPath:
                path_length = 0
                
            dual_VRP[i][j] = path_length + df.iloc[j]['demand']
    
    # Instantiate QUBO
    qubo_size = len(df) * num_vehicles * time_capacity
    qubo_2D = np.zeros((qubo_size, qubo_size))
    
    def map_to_index(arc, veh, time):
        return arc * (num_vehicles * time_capacity) + veh * time_capacity + time
    
    # Iterate through all indices
    for arc_1, arc_2, veh_1, veh_2, time_1, time_2 in itertools.product(
        range(len(df)), range(len(df)), 
        range(num_vehicles), range(num_vehicles),
        range(time_capacity), range(time_capacity)
    ):
        idx1 = map_to_index(arc_1, veh_1, time_1)
        idx2 = map_to_index(arc_2, veh_2, time_2)
                
        # Visit each arc only once
        if arc_1 == arc_2 and time_1 != time_2:
            qubo_2D[idx1, idx2] = lambd
        # No duplicating vehicles
        elif arc_1 != arc_2 and time_1 == time_2 and veh_1 == veh_2:
            qubo_2D[idx1, idx2] = lambd
        # No collisions
        elif arc_1 == arc_2 and time_1 == time_2 and veh_1 != veh_2:
            qubo_2D[idx1, idx2] = lambd
            
        # Route arc to all neighbors for given vehicle:
        if arc_1 != arc_2 and time_1 != time_2 and veh_1 == veh_2:
            if dual_VRP[arc_1, arc_2] == time_2 - time_1:
                # Fixed comparison operator (was using == instead of =)
                qubo_2D[idx1, idx2] = mu
                
    return dual_VRP, qubo_2D
    
if __name__ == "__main__":

    # Process all files from S1 to S10
    for i in range(1, 11):
        file_path = f"data/TCARPdata/TCARP-S{i}.dat"
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping...")
            continue
            
        print(f"Processing TCARP-S{i}.dat...")
        
        # Open and process the file
        with open(file_path, "r") as file:
            dual_VRP, qubo_2D = preprocess_tcarp(file)
            
            # Save dual_VRP to file
            dual_vrp_output = f"data/TCARPdata/preprocessed/dual_vrp/dual_VRP-S{i}.npy"
            np.save(dual_vrp_output, dual_VRP)
            print(f"  Saved {dual_vrp_output}")
            
            # Save qubo_2D to file
            qubo_output = f"data/TCARPdata/preprocessed/qubo/qubo_2D-S{i}.npy"
            np.save(qubo_output, qubo_2D)
            print(f"  Saved {qubo_output}")
            
            # Print stats
            print(f"  QUBO matrix shape: {qubo_2D.shape}")
            print(f"  Sparsity: {(1-np.count_nonzero(qubo_2D)/qubo_2D.size)*100:.2f}%")
            
    print("Processing complete!")