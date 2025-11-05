"""
A* Pathfinding Algorithm Implementation for City Route Finding
"""

import math
import heapq
import matplotlib.pyplot as plt

# City coordinates
coords = {
    'A': (0, 0),
    'B': (2, 1),
    'C': (1, 4),
    'D': (4, 2),
    'E': (5, 4),
    'F': (6, 1)
}

# Graph edges with distances
graph = {
    'A': {'B': 2.2, 'C': 4.1},
    'B': {'A': 2.2, 'D': 2.2, 'E': 4.2},
    'C': {'A': 4.1, 'E': 4.3},
    'D': {'B': 2.2, 'E': 2.2, 'F': 2.2},
    'E': {'B': 4.2, 'C': 4.3, 'D': 2.2, 'F': 1.8},
    'F': {'D': 2.2, 'E': 1.8}
}

def heuristic(a, b):
    """
    Calculate Euclidean distance heuristic between two nodes
    
    Args:
        a (str): Starting node
        b (str): Target node
        
    Returns:
        float: Euclidean distance between nodes
    """
    (x1, y1) = coords[a]
    (x2, y2) = coords[b]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star(start, goal):
    """
    Find shortest path using A* algorithm
    
    Args:
        start (str): Starting node
        goal (str): Target node
        
    Returns:
        tuple: (path, cost) if path exists, (None, infinity) otherwise
    """
    # Initialize data structures
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)
    
    # A* search
    while open_list:
        current = heapq.heappop(open_list)[1]
        
        # Goal reached
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal]
        
        # Explore neighbors
        for neighbor, cost in graph[current].items():
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
                
    return None, float('inf')

def draw_graph(path=None):
    """
    Visualize the graph and path
    
    Args:
        path (list, optional): Path to highlight
    """
    plt.figure(figsize=(8, 6))
    drawn_edges = set()
    
    # Draw edges and distances
    for city, neighbors in graph.items():
        for neighbor, dist in neighbors.items():
            if (neighbor, city) not in drawn_edges:
                drawn_edges.add((city, neighbor))
                x1, y1 = coords[city]
                x2, y2 = coords[neighbor]
                plt.plot([x1, x2], [y1, y2], 'gray', linewidth=1)
                midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
                plt.text(midx, midy, f"{dist:.1f}", fontsize=8, color='red')

    # Plot cities
    for city, (x, y) in coords.items():
        plt.scatter(x, y, s=300, color='skyblue', edgecolors='black', zorder=3)
        plt.text(x + 0.1, y + 0.1, city, fontsize=12, weight='bold')

    # Highlight path
    if path:
        px = [coords[p][0] for p in path]
        py = [coords[p][1] for p in path]
        plt.plot(px, py, color='orange', linewidth=3, label='Shortest Path')
        plt.scatter(px[0], py[0], s=150, color='green', label='Start', zorder=4)
        plt.scatter(px[-1], py[-1], s=150, color='red', label='Goal', zorder=4)

    plt.legend()
    plt.grid(True)
    plt.title("City Map - A* Path Finder")
    plt.axis('equal')
    plt.show()

def get_user_input():
    """
    Get start and goal cities from user with validation
    
    Returns:
        tuple: (start_city, goal_city) or (None, None) if invalid
    """
    available_cities = ', '.join(sorted(coords.keys()))
    print(f"Available cities: {available_cities}")
    print("Example: Start = A, Goal = F")
    
    start = input("Enter start city (default A): ").upper() or "A"
    goal = input("Enter goal city (default F): ").upper() or "F"
    
    # Validate input
    if start not in graph:
        print(f"❌ Invalid start city '{start}'. Please choose from: {available_cities}")
        return None, None
        
    if goal not in graph:
        print(f"❌ Invalid goal city '{goal}'. Please choose from: {available_cities}")
        return None, None
        
    return start, goal

def main():
    """Main function to run the path finder"""
    start, goal = get_user_input()
    
    if start and goal:
        path, cost = a_star(start, goal)
        if path:
            print(f"\n✅ Shortest path: {' → '.join(path)}")
            print(f"🛣️ Total distance: {cost:.2f}")
            draw_graph(path)
        else:
            print("❌ No path found!")

if __name__ == "__main__":
    main()