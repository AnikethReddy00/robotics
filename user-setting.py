import numpy as np
import heapq
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Grid & Drone Setup
GRID_SIZE = 20
NUM_OBSTACLES = 30
DRONE_MIN_HEIGHT = 1
DRONE_MAX_HEIGHT = 5

# Generate obstacles (crowds) on the ground (z=0)
def generate_obstacles(num):
    obstacles = set()
    while len(obstacles) < num:
        x, y = random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)
        obstacles.add((x, y, 0))
    return obstacles

# A* Path Planning (3D)
class AStar3D:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def get_neighbors(self, node):
        x, y, z = node
        moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in moves]
        return [n for n in neighbors if 0 <= n[0] < GRID_SIZE and 0 <= n[1] < GRID_SIZE and DRONE_MIN_HEIGHT <= n[2] <= DRONE_MAX_HEIGHT]

    def is_valid(self, node):
        return node not in self.obstacles

    def find_path(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if not self.is_valid(neighbor):
                    continue
                temp_g_score = g_score[current] + 1
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        return []

# Move obstacles dynamically
def move_obstacles(obstacles):
    new_obstacles = set()
    for x, y, z in obstacles:
        move_x, move_y = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
        new_x, new_y = max(1, min(GRID_SIZE - 2, x + move_x)), max(1, min(GRID_SIZE - 2, y + move_y))
        new_obstacles.add((new_x, new_y, 0))
    return new_obstacles

# 3D Visualization
def visualize_3d(start, goal, obstacles):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_zlim(0, DRONE_MAX_HEIGHT + 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')

    plt.ion()  # Interactive mode

    drone_position = start
    encountered_crowds = set()
    previous_paths = []

    while drone_position != goal:
        obstacles = move_obstacles(obstacles)
        obstacles.discard(start)
        obstacles.discard(goal)

        path = AStar3D(drone_position, goal, obstacles).find_path()
        if not path:
            print("No valid path found. Stopping simulation.")
            break

        # Save the old path for visualization
        previous_paths.append(path)

        ax.cla()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_zlim(0, DRONE_MAX_HEIGHT + 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')

        # Draw previous paths in light gray
        for prev_path in previous_paths[:-1]:
            path_x, path_y, path_z = zip(*prev_path)
            ax.plot(path_x, path_y, path_z, 'gray', linestyle='dotted', linewidth=1)

        # Draw obstacles
        for (x, y, z) in obstacles:
            ax.bar3d(x, y, 0, 0.2, 0.2, 1, color='red', alpha=0.8, edgecolor='black')

        # Draw current path
        if path:
            path_x, path_y, path_z = zip(*path)
            ax.plot(path_x, path_y, path_z, 'g-', linewidth=2, label="Current Path")

        # Start & Goal
        ax.scatter(*start, c='yellow', marker='o', s=100, label="Start")
        ax.scatter(*goal, c='black', marker='o', s=100, label="Goal")

        # Move drone
        next_position = path[0]
        if (next_position[0], next_position[1], 0) in obstacles:
            print(f"⚠️ Warning: Drone encountered a crowd at {next_position}! Finding alternative route...")

            # Recalculate path without changing altitude
            new_path = AStar3D(drone_position, goal, obstacles).find_path()
            if new_path:
                print(f"🔄 Rerouting on the same level: {new_path}")
                new_path_x, new_path_y, new_path_z = zip(*new_path)
                ax.plot(new_path_x, new_path_y, new_path_z, 'cyan', linewidth=2, linestyle='dashed', label="Rerouted Path")
                next_position = new_path[0]

        # Draw crowd encounters
        if next_position in obstacles:
            encountered_crowds.add(next_position)
            ax.scatter(*next_position, c='purple', marker='o', s=150, label="Crowd Encounter")

        drone_position = next_position
        ax.scatter(*drone_position, c='blue', marker='o', s=100, label="Drone")

        plt.draw()
        plt.pause(0.5)

    plt.ioff()
    plt.show(block=True)

# User Input for Start & Goal
def get_user_input():
    while True:
        try:
            start_x, start_y = map(int, input("Enter start coordinates (x y): ").split())
            goal_x, goal_y = map(int, input("Enter goal coordinates (x y): ").split())

            if 0 <= start_x < GRID_SIZE and 0 <= start_y < GRID_SIZE and 0 <= goal_x < GRID_SIZE and 0 <= goal_y < GRID_SIZE:
                return (start_x, start_y, DRONE_MIN_HEIGHT), (goal_x, goal_y, DRONE_MIN_HEIGHT)
            else:
                print("Invalid input! Coordinates must be within grid range.")
        except ValueError:
            print("Invalid input! Enter two integers separated by a space.")

# Run Simulation
def drone_navigation():
    start, goal = get_user_input()
    obstacles = generate_obstacles(NUM_OBSTACLES)
    obstacles.discard(start)
    obstacles.discard(goal)

    visualize_3d(start, goal, obstacles)

drone_navigation()
