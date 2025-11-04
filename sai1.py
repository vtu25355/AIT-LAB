from collections import deque

def max_profit_bfs_companies(prices):
    n = len(prices)
    companies = list(prices[0].keys())
    
    # Queue stores (day_index, holding_company_or_None, profit, transaction_history)
    queue = deque()
    queue.append((0, None, 0, []))  # start: day 0, no stock, profit 0, empty history
    max_profit = 0
    best_history = []
    
    visited = set()  # (day, holding_company, profit)
    
    while queue:
        day, holding, profit, history = queue.popleft()
        if day == n:
            if profit > max_profit:
                max_profit = profit
                best_history = history
            continue
        
        # Skip day
        state = (day + 1, holding, profit)
        if state not in visited:
            visited.add(state)
            queue.append((day + 1, holding, profit, history.copy()))
        
        # Buy a stock (if not holding any)
        if holding is None:
            for company in companies:
                new_profit = profit - prices[day][company]
                new_history = history.copy()
                new_history.append(f"Day {day}: Buy {company} at {prices[day][company]}")
                state = (day + 1, company, new_profit)
                if state not in visited:
                    visited.add(state)
                    queue.append((day + 1, company, new_profit, new_history))
        
        # Sell the stock (if holding one)
        if holding is not None:
            new_profit = profit + prices[day][holding]
            new_history = history.copy()
            new_history.append(f"Day {day}: Sell {holding} at {prices[day][holding]}")
            state = (day + 1, None, new_profit)
            if state not in visited:
                visited.add(state)
                queue.append((day + 1, None, new_profit, new_history))
    
    return max_profit, best_history

# Example usage
prices = [
    {"AAPL": 150, "GOOG": 2700, "MSFT": 300},
    {"AAPL": 155, "GOOG": 2720, "MSFT": 295},
    {"AAPL": 148, "GOOG": 2750, "MSFT": 310},
    {"AAPL": 160, "GOOG": 2800, "MSFT": 320}
]

profit, transactions = max_profit_bfs_companies(prices)
print("Maximum Profit:", profit)
print("\nTransaction Sequence:")
for t in transactions:
    print(t)

