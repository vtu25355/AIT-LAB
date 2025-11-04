import random

# Places
places = ["Depot", "A", "B", "C", "D"]

# Distance adjacency matrix
# matrix[i][j] = distance from place i to j
matrix = [
    [0, 2, 5, 6, 8],  # Depot
    [2, 0, 4, 7, 3],  # A
    [5, 4, 0, 2, 6],  # B
    [6, 7, 2, 0, 5],  # C
    [8, 3, 6, 5, 0]   # D
]

# Calculate total distance of a route
def total_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        from_idx = places.index(route[i])
        to_idx = places.index(route[i + 1])
        dist += matrix[from_idx][to_idx]
    # Return to depot
    dist += matrix[places.index(route[-1])][places.index(route[0])]
    return dist

# Generate neighbor by swapping two locations (excluding depot)
def generate_neighbor(route):
    new_route = route.copy()
    i, j = random.sample(range(1, len(route)), 2)  # avoid depot at index 0
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# Hill Climbing Algorithm
def hill_climbing(places):
    # Initial route: start at depot, visit all other locations
    route = ["Depot"] + [p for p in places if p != "Depot"]
    random.shuffle(route[1:])
    
    current_distance = total_distance(route)
    iterations = 0
    
    while True:
        neighbor = generate_neighbor(route)
        neighbor_distance = total_distance(neighbor)
        if neighbor_distance < current_distance:
            route = neighbor
            current_distance = neighbor_distance
            iterations += 1
        else:
            break  # No better neighbor found (local optimum)
    
    return route, current_distance, iterations

# Run the hill climbing VRP
best_route, best_distance, iters = hill_climbing(places)

# Output results
print("Best route found:", best_route)
print("Total distance:", best_distance)
print("Iterations:", iters)

