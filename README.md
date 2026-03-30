# Sokoban-Game-Using-Chebyshev-Distance

🚀 Features
Dual Gameplay Modes: Switch between manual play using arrow keys and an automated AI solver.

Advanced A Algorithm*: Implements a custom search logic inheriting from a base A* class to find optimal solutions.

Heuristic Optimization: Uses a sophisticated heuristic combining Chebyshev distance and Minimum Spanning Tree (MST) to minimize node expansion.

Dynamic Visuals: Features a detailed graphical interface with custom-drawn characters, animated movement, and a real-time statistics panel.

Performance Logging: Automatically tracks and prints experiment logs including time taken, nodes expanded, and path costs after each run.

🛠️ Requirements
Python 3.x

Pygame Library

Install dependencies via pip:

Note: Ensure astar_base.py is present in the same directory as the main script.

🎮 How to Play
1. Map Configuration
The game loads maps from a .txt file (default: example_map.txt). Use the following legend:

% : Wall

A : Player Start Position

B : Box

D : Target (Goal)

C : Box already on Target

2. Controls
🧠 Technical Details
State Representation
The search state is defined as a tuple containing the player's coordinates and a frozenset of box positions. This ensures states are hashable and can be efficiently tracked in the "visited" set.

Heuristic Logic
To solve complex puzzles, the solver calculates the estimated cost to reach the goal by:

Building a Minimum Spanning Tree of the boxes to understand their spatial distribution.

Calculating the minimum distance from each box to its nearest target.

Summing these values to provide a consistent and admissible heuristic for the A* search.

📊 Statistics & Experiment Log
Upon exiting the game, the program outputs a detailed performance table to the terminal:

Time(s): CPU time spent calculating the path.

Nodes: Total states explored before finding the solution.

Cost: Total number of moves in the generated solution.
