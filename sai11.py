def weather_dfs(day, n, current_weather, path, all_paths, transitions):
    """
    Recursive DFS to explore weather sequences.
    
    day: current day index
    n: total days to forecast
    current_weather: current weather state
    path: current sequence of weather
    all_paths: list of all possible sequences
    transitions: dict of possible weather transitions
    """
    path.append(current_weather)
    
    if day == n - 1:
        all_paths.append(path.copy())
    else:
        for next_weather in transitions[current_weather]:
            weather_dfs(day + 1, n, next_weather, path, all_paths, transitions)
    
    path.pop()  # backtrack

# Example usage
if __name__ == "__main__":
    n_days = 3
    initial_weather = "Sunny"
    
    # Possible transitions
    transitions = {
        "Sunny": ["Sunny", "Cloudy", "Rainy"],
        "Cloudy": ["Sunny", "Cloudy", "Rainy"],
        "Rainy": ["Cloudy", "Rainy"]
    }
    
    all_sequences = []
    weather_dfs(0, n_days, initial_weather, [], all_sequences, transitions)
    
    print(f"All possible weather sequences for {n_days} days starting with {initial_weather}:")
    for seq in all_sequences:
        print(seq)
