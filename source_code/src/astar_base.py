import math
from heapq import heappush, heappop

class A_Star:
    
    def get_successors(self, state):
        raise NotImplementedError

    def heuristic(self, state):
        raise NotImplementedError

    def is_goal(self, state):
        raise NotImplementedError

    def state_to_key(self, state):
        return str(state)

    def solve(self, initial_state):
        h0      = self.heuristic(initial_state)
        key0    = self.state_to_key(initial_state)
        # (f, g, counter, state, parent_key, action, path_so_far)
        counter = 0
        frontier = [(h0, 0, counter, initial_state, None, None)]
        best_g   = {key0: 0.0}
        came_from = {}         
        nodes_expanded = 0

        while frontier:
            f, g, _, curr_state, _, _ = frontier[0]
            heappop(frontier)

            curr_key = self.state_to_key(curr_state)

            if g > best_g.get(curr_key, math.inf):
                continue

            nodes_expanded += 1

            if self.is_goal(curr_state):
                path, actions = [], []
                k = curr_key
                while k in came_from:
                    parent_k, action, state = came_from[k]
                    path.append(state)
                    actions.append(action)
                    k = parent_k
                    
                path.append(initial_state)
                path.reverse()
                actions.reverse()
                return path, actions, nodes_expanded

            for action, next_state, cost in self.get_successors(curr_state):
                next_key = self.state_to_key(next_state)
                new_g    = g + cost
                if new_g < best_g.get(next_key, math.inf):
                    best_g[next_key]    = new_g
                    came_from[next_key] = (curr_key, action, next_state)
                    new_h    = self.heuristic(next_state)
                    counter += 1
                    heappush(frontier, (new_g + new_h, new_g, counter, next_state, curr_key, action))

        return [], [], nodes_expanded
