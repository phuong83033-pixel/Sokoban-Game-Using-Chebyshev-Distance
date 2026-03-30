import heapq
import pygame
import sys
import time
from astar_base import A_Star   

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

TILE = 64
FPS  = 60

DIRS = [
    ((-1,  0), 'North'),
    (( 1,  0), 'South'),
    (( 0, -1), 'West'),
    (( 0,  1), 'East'),
]

COLOR = {
    'bg':          (245, 245, 220),
    'wall':        ( 80,  60,  40),
    'wall_light':  (110,  85,  60),
    'floor':       (210, 195, 160),
    'floor_alt':   (200, 183, 148),
    'box':         (180,  90,  20),
    'box_border':  (120,  60,  10),
    'box_done':    ( 60, 160,  60),
    'box_done_bd': ( 30, 110,  30),
    'target':      (220,  60,  60),
    'player':      ( 50, 120, 210),
    'player_bd':   ( 25,  70, 140),
    'white':       (255, 255, 255),
    'black':       (  0,   0,   0),
    'panel':       ( 35,  35,  55),
    'panel_light': ( 55,  55,  80),
    'yellow':      (255, 215,  50),
    'green':       ( 60, 200,  80),
    'red':         (220,  60,  60),
    'blue':        ( 60, 140, 220),
    'text':        (210, 210, 225),
    'text_dim':    (140, 140, 160),
}


# ─────────────────────────────────────────────────────────────
#  CLASS: Maze
# ─────────────────────────────────────────────────────────────

class Maze:
    def __init__(self, filename):
        self.filename     = filename
        self.walls        = set()
        self.targets      = set()
        self.boxes        = set()
        self.player_start = None
        self.rows         = 0
        self.cols         = 0
        self._load()

    def _load(self):
        with open(self.filename, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        self.rows = len(lines)
        self.cols = max((len(l) for l in lines), default=0)
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if   ch == '%': self.walls.add((r, c))
                elif ch == 'A': self.player_start = (r, c)
                elif ch == 'B': self.boxes.add((r, c))
                elif ch == 'D': self.targets.add((r, c))
                elif ch == 'C':
                    self.boxes.add((r, c))
                    self.targets.add((r, c))
        if self.player_start is None:
            raise ValueError("[ERROR] Map missing character 'A'")


# ─────────────────────────────────────────────────────────────
#  CLASS: SokobanAStar  (kế thừa A_Star base)
# ─────────────────────────────────────────────────────────────

class SokobanAStar(A_Star):
    def __init__(self, maze: Maze):
        self.maze = maze

    # ── override ────────────────────────────────────────────────

    def is_goal(self, state):
        _, boxes = state
        return self.maze.targets == set(boxes)

    def heuristic(self, state):
        _, boxes = state
        return self._heuristic_mst(boxes, self.maze.targets)

    def get_successors(self, state):
        player, boxes = state
        successors = []
        for (dr, dc), direction in DIRS:
            result = self._get_next_state(player, boxes, dr, dc)
            if result is not None:
                new_player, new_boxes = result
                successors.append((direction, (new_player, new_boxes), 1))
        return successors

    def state_to_key(self, state):
        player, boxes = state
        return (player, boxes)   

    # ── helpers (private) ──────────────────────────────────────

    def _get_next_state(self, player, boxes, dr, dc):
        nr, nc = player[0] + dr, player[1] + dc
        if (nr, nc) in self.maze.walls:
            return None
        new_boxes = set(boxes)
        if (nr, nc) in new_boxes:
            nnr, nnc = nr + dr, nc + dc
            if (nnr, nnc) in self.maze.walls or (nnr, nnc) in new_boxes:
                return None
            new_boxes.discard((nr, nc))
            new_boxes.add((nnr, nnc))
        return (nr, nc), frozenset(new_boxes)

    @staticmethod
    def _chebyshev(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def _heuristic_mst(self, boxes, targets):
        box_list = list(boxes)
        tgt_list = list(targets)
        n = len(box_list)
        if n == 0:
            return 0
        min_dist = [min(self._chebyshev(b, t) for t in tgt_list) for b in box_list]
        if n == 1:
            return min_dist[0]
        in_mst = [False] * n
        key    = [10**9] * n
        key[0] = 0
        total  = 0
        for _ in range(n):
            u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
            in_mst[u] = True
            total += key[u]
            for v in range(n):
                if not in_mst[v]:
                    w = self._chebyshev(box_list[u], box_list[v])
                    if w < key[v]:
                        key[v] = w
        return total + sum(min_dist)

    # ── public solve wrapper ────────────────────────────────────

    def solve(self):
        initial = (self.maze.player_start, frozenset(self.maze.boxes))
        path, actions, nodes = super().solve(initial)
        cost = len(actions) if actions else -1
        return (actions if actions else None), cost, nodes


# ─────────────────────────────────────────────────────────────
#  CLASS: Character
# ─────────────────────────────────────────────────────────────

class Character:
    def __init__(self, start_pos):
        self.pos = start_pos

    def reset(self, start_pos):
        self.pos = start_pos


# ─────────────────────────────────────────────────────────────
#  CLASS: Game
# ─────────────────────────────────────────────────────────────

class Game:
    PANEL_W = 270

    def __init__(self, map_file):
        pygame.init()
        pygame.display.set_caption("Sokoban - AIMidterm")
        self.maze      = Maze(map_file)
        self.character = Character(self.maze.player_start)
        self.boxes     = set(self.maze.boxes)
        self.font_title = pygame.font.SysFont('consolas', 20, bold=True)
        self.font_med   = pygame.font.SysFont('consolas', 15)
        self.font_sm    = pygame.font.SysFont('consolas', 13)
        win_w = self.maze.cols * TILE + self.PANEL_W
        win_h = max(self.maze.rows * TILE, 500)
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.clock  = pygame.time.Clock()
        self.mode         = 'manual'
        self.move_count   = 0
        self.solved       = False
        self.message      = "Use arrow keys to play  |  SPACE = Auto solve"
        self.msg_color    = COLOR['text']
        self.solution     = []
        self.solution_idx = 0
        self.auto_delay   = 0.30
        self.last_auto_t  = 0.0
        self.solve_time   = 0.0
        self.nodes_exp    = 0
        self.total_cost   = 0
        self.stats_log    = []

    def _reset(self):
        self.character.reset(self.maze.player_start)
        self.boxes        = set(self.maze.boxes)
        self.move_count   = 0
        self.solved       = False
        self.solution     = []
        self.solution_idx = 0
        self.mode         = 'manual'
        self.message      = "Reset done  |  SPACE = Auto solve"
        self.msg_color    = COLOR['text']

    def _do_move(self, dr, dc):
        if self.solved:
            return
        solver = SokobanAStar(self.maze)
        result = solver._get_next_state(
            self.character.pos, frozenset(self.boxes), dr, dc
        )
        if result is None:
            return
        self.character.pos, new_boxes = result
        self.boxes      = set(new_boxes)
        self.move_count += 1
        if self.maze.targets == frozenset(self.boxes):
            self.solved    = True
            self.mode      = 'manual'
            self.message   = "SOLVED! " + str(self.move_count) + " moves"
            self.msg_color = COLOR['green']

    def _start_auto(self):
        self.character.reset(self.maze.player_start)
        self.boxes        = set(self.maze.boxes)
        self.move_count   = 0
        self.solved       = False
        self.solution     = []
        self.solution_idx = 0
        self.message      = "A* is solving..."
        self.msg_color    = COLOR['yellow']
        self._render()
        pygame.display.flip()

        solver  = SokobanAStar(self.maze)   
        t0      = time.perf_counter()
        actions, cost, nodes = solver.solve()
        elapsed = time.perf_counter() - t0

        self.solve_time = elapsed
        self.nodes_exp  = nodes
        self.total_cost = cost if cost >= 0 else 0
        self.stats_log.append({
            'time_s':    round(elapsed, 4),
            'nodes':     nodes,
            'path_cost': cost,
        })

        if actions is None:
            self.message   = "No solution found!"
            self.msg_color = COLOR['red']
            self.mode      = 'manual'
        else:
            self.solution     = actions
            self.solution_idx = 0
            self.mode         = 'playing'
            self.last_auto_t  = time.perf_counter()
            self.message      = ("Solved! " + str(cost) + " moves | "
                                 + str(nodes) + " nodes | "
                                 + str(round(elapsed, 3)) + "s  Replaying...")
            self.msg_color    = COLOR['green']

    def _auto_step(self):
        if self.solution_idx >= len(self.solution):
            self.solved    = True
            self.mode      = 'manual'
            self.message   = ("SOLVED! " + str(self.total_cost)
                              + " moves | " + str(self.nodes_exp)
                              + " nodes | " + str(round(self.solve_time, 3)) + "s")
            self.msg_color = COLOR['green']
            return
        direction = self.solution[self.solution_idx]
        dr, dc    = next(d for d, n in DIRS if n == direction)
        self._do_move(dr, dc)
        self.solution_idx += 1

    # ── Draw helpers ──────────────────────────────────────────

    def _draw_wall(self, r, c):
        x, y = c*TILE, r*TILE
        pygame.draw.rect(self.screen, COLOR['wall'], (x, y, TILE, TILE))
        for ri in range(3):
            shift = (TILE//4) if ri % 2 else 0
            ry = y + ri*(TILE//3)
            for ci in range(-1, 3):
                bx = x + ci*(TILE//2) + shift + 2
                pygame.draw.rect(self.screen, COLOR['wall_light'],
                                 (bx, ry+2, TILE//2-4, TILE//3-4), 1)

    def _draw_target(self, r, c):
        cx = c*TILE + TILE//2
        cy = r*TILE + TILE//2
        pygame.draw.circle(self.screen, COLOR['target'], (cx, cy), TILE//7)
        pygame.draw.circle(self.screen, (160, 30, 30),  (cx, cy), TILE//7, 2)

    def _draw_box(self, r, c, on_target):
        pad    = 7
        col    = COLOR['box_done']    if on_target else COLOR['box']
        border = COLOR['box_done_bd'] if on_target else COLOR['box_border']
        rect   = pygame.Rect(c*TILE+pad, r*TILE+pad, TILE-2*pad, TILE-2*pad)
        pygame.draw.rect(self.screen, col,    rect, border_radius=6)
        pygame.draw.rect(self.screen, border, rect, 3, border_radius=6)
        p = pad + 6
        pygame.draw.line(self.screen, border,
                         (c*TILE+p, r*TILE+p), (c*TILE+TILE-p, r*TILE+TILE-p), 2)
        pygame.draw.line(self.screen, border,
                         (c*TILE+TILE-p, r*TILE+p), (c*TILE+p, r*TILE+TILE-p), 2)

    def _draw_player(self, r, c):
        x0 = c * TILE
        y0 = r * TILE
        SKIN      = (255, 200, 150)
        SKIN_DARK = (220, 160, 100)
        HAIR      = ( 60,  35,  10)
        SHIRT     = ( 50, 120, 210)
        SHIRT_SH  = ( 30,  80, 160)
        PANTS     = ( 40,  60, 130)
        SHOES     = ( 40,  30,  20)
        EYE_W     = (255, 255, 255)
        EYE_B     = ( 20,  20,  20)
        MOUTH     = (180,  80,  80)

        head_r  = TILE // 7 + 2
        head_cx = x0 + TILE // 2
        head_cy = y0 + TILE // 5 + head_r
        neck_y   = head_cy + head_r
        body_top = neck_y + 2
        body_bot = y0 + TILE * 62 // 100
        body_w_t = TILE // 5
        body_w_b = TILE // 6
        leg_h    = TILE * 28 // 100
        leg_w    = TILE // 8
        leg_gap  = TILE // 12
        leg_top  = body_bot
        leg_bot  = leg_top + leg_h
        shoe_h   = TILE // 12
        shoe_ext = TILE // 10
        lx = head_cx - leg_gap - leg_w
        pygame.draw.rect(self.screen, PANTS, (lx, leg_top, leg_w, leg_h), border_radius=3)
        rx = head_cx + leg_gap
        pygame.draw.rect(self.screen, PANTS, (rx, leg_top, leg_w, leg_h), border_radius=3)
        pygame.draw.rect(self.screen, SHOES,
                         (lx - shoe_ext//2, leg_bot - shoe_h, leg_w + shoe_ext, shoe_h + 2), border_radius=2)
        pygame.draw.rect(self.screen, SHOES,
                         (rx - shoe_ext//2, leg_bot - shoe_h, leg_w + shoe_ext, shoe_h + 2), border_radius=2)
        body_pts = [
            (head_cx - body_w_t, body_top),
            (head_cx + body_w_t, body_top),
            (head_cx + body_w_b, body_bot),
            (head_cx - body_w_b, body_bot),
        ]
        pygame.draw.polygon(self.screen, SHIRT, body_pts)
        pygame.draw.polygon(self.screen, SHIRT_SH, body_pts, 2)
        pygame.draw.line(self.screen, SHIRT_SH,
                         (head_cx, body_top + 2), (head_cx, body_bot - 2), 1)
        arm_w   = TILE // 10
        arm_h   = (body_bot - body_top) * 7 // 10
        arm_top = body_top + 2
        larm_x = head_cx - body_w_t - arm_w + 1
        pygame.draw.rect(self.screen, SHIRT, (larm_x, arm_top, arm_w, arm_h), border_radius=3)
        pygame.draw.circle(self.screen, SKIN, (larm_x + arm_w//2, arm_top + arm_h), arm_w//2 + 1)
        rarm_x = head_cx + body_w_t - 1
        pygame.draw.rect(self.screen, SHIRT, (rarm_x, arm_top, arm_w, arm_h), border_radius=3)
        pygame.draw.circle(self.screen, SKIN, (rarm_x + arm_w//2, arm_top + arm_h), arm_w//2 + 1)
        pygame.draw.circle(self.screen, SKIN,      (head_cx, head_cy), head_r)
        pygame.draw.circle(self.screen, SKIN_DARK, (head_cx, head_cy), head_r, 1)
        hair_rect = pygame.Rect(head_cx - head_r, head_cy - head_r, head_r * 2, head_r + 2)
        pygame.draw.ellipse(self.screen, HAIR, hair_rect)
        pygame.draw.rect(self.screen, HAIR,
                         (head_cx - head_r + 1, head_cy - 2, head_r * 2 - 2, 4), border_radius=2)
        eye_y  = head_cy + 1
        eye_off = head_r // 3
        for ex in [head_cx - eye_off, head_cx + eye_off]:
            pygame.draw.circle(self.screen, EYE_W, (ex, eye_y), head_r // 4)
            pygame.draw.circle(self.screen, EYE_B, (ex, eye_y), head_r // 7)
        mouth_rect = pygame.Rect(head_cx - head_r//3, eye_y + head_r//3,
                                 head_r * 2 // 3, head_r // 3)
        pygame.draw.arc(self.screen, MOUTH, mouth_rect, 3.14, 0, 2)
        for ex, sign in [(head_cx - head_r, -1), (head_cx + head_r, 1)]:
            ear_rect = pygame.Rect(ex - 2 + sign, head_cy - head_r//3, 4, head_r * 2 // 3)
            pygame.draw.ellipse(self.screen, SKIN,      ear_rect)
            pygame.draw.ellipse(self.screen, SKIN_DARK, ear_rect, 1)

    def _render_map(self):
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                col = COLOR['floor'] if (r+c)%2==0 else COLOR['floor_alt']
                pygame.draw.rect(self.screen, col, (c*TILE, r*TILE, TILE, TILE))
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if   (r,c) in self.maze.walls:   self._draw_wall(r, c)
                elif (r,c) in self.maze.targets:  self._draw_target(r, c)
        for (r,c) in self.boxes:
            self._draw_box(r, c, on_target=(r,c) in self.maze.targets)
        pr, pc = self.character.pos
        self._draw_player(pr, pc)

    def _render_panel(self):
        pw = self.PANEL_W
        ph = self.screen.get_height()
        px = self.maze.cols * TILE
        pygame.draw.rect(self.screen, COLOR['panel'], (px, 0, pw, ph))
        pygame.draw.line(self.screen, COLOR['panel_light'], (px,0), (px,ph), 2)
        y = 12
        t = self.font_title.render("SOKOBAN", True, COLOR['yellow'])
        self.screen.blit(t, (px+15, y)); y += 30
        if self.mode == 'playing':
            bcol, btxt = COLOR['green'], ">> AUTO  (A* Replaying)"
        else:
            bcol, btxt = COLOR['blue'],  "   MANUAL  (Arrow Keys)"
        pygame.draw.rect(self.screen, bcol, (px+10, y, pw-20, 28), border_radius=6)
        self.screen.blit(self.font_med.render(btxt, True, COLOR['white']), (px+14, y+5))
        y += 40

        def stat(label, val, vc=COLOR['text']):
            nonlocal y
            self.screen.blit(self.font_sm.render(label, True, COLOR['text_dim']), (px+15, y))
            self.screen.blit(self.font_sm.render(str(val), True, vc), (px+pw//2, y))
            y += 19

        stat("Moves :",      self.move_count)
        boxes_on = sum(1 for b in self.boxes if b in self.maze.targets)
        stat("Boxes done :", f"{boxes_on}/{len(self.maze.targets)}")
        y += 6
        pygame.draw.line(self.screen, COLOR['panel_light'], (px+10,y), (px+pw-10,y), 1)
        y += 8
        self.screen.blit(self.font_sm.render("-- A* Results --", True, COLOR['text_dim']), (px+15, y)); y += 18
        stat("Nodes :",   self.nodes_exp)
        stat("Cost :",    self.total_cost)
        stat("Time :",    f"{self.solve_time:.3f}s")
        if self.mode == 'playing' and self.solution:
            remaining = len(self.solution) - self.solution_idx
            stat("Remaining :", f"{remaining} steps", COLOR['yellow'])
        y += 6
        pygame.draw.line(self.screen, COLOR['panel_light'], (px+10,y), (px+pw-10,y), 1)
        y += 8
        self.screen.blit(self.font_sm.render("-- Controls --", True, COLOR['text_dim']), (px+15, y)); y += 18
        helps = [("Arrows", "Move player"), ("SPACE", "A* auto-solve"), ("R", "Reset"), ("Q", "Quit")]
        for k, d in helps:
            self.screen.blit(self.font_sm.render(k, True, COLOR['yellow']), (px+15, y))
            self.screen.blit(self.font_sm.render(d, True, COLOR['text_dim']), (px+85, y))
            y += 17
        msg_y = ph - 75
        pygame.draw.rect(self.screen, COLOR['panel_light'], (px+5, msg_y-5, pw-10, 68), border_radius=5)
        words = self.message.split()
        line  = ""
        cy2   = msg_y
        for w in words:
            test = (line + " " + w).strip()
            if self.font_sm.size(test)[0] < pw - 18:
                line = test
            else:
                self.screen.blit(self.font_sm.render(line, True, self.msg_color), (px+10, cy2))
                cy2 += 16; line = w
        if line:
            self.screen.blit(self.font_sm.render(line, True, self.msg_color), (px+10, cy2))

    def _render(self):
        self.screen.fill(COLOR['bg'])
        self._render_map()
        self._render_panel()

    def run(self):
        DIR_MAP = {
            pygame.K_UP:    (-1, 0),
            pygame.K_DOWN:  ( 1, 0),
            pygame.K_LEFT:  ( 0,-1),
            pygame.K_RIGHT: ( 0, 1),
        }
        running = True
        while running:
            self.clock.tick(FPS)
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_q:
                        running = False
                    elif k == pygame.K_r:
                        self._reset()
                    elif k == pygame.K_SPACE:
                        self._start_auto()
                        self.last_auto_t = time.perf_counter()
                    elif self.mode == 'manual' and k in DIR_MAP:
                        self._do_move(*DIR_MAP[k])
            if (self.mode == 'playing'
                    and self.solution
                    and not self.solved
                    and now - self.last_auto_t >= self.auto_delay):
                self._auto_step()
                self.last_auto_t = now
            self._render()
            pygame.display.flip()
        pygame.quit()
        self._print_stats()
        sys.exit()

    def _print_stats(self):
        if not self.stats_log:
            return
        print("\n" + "="*50)
        print("  A* Experiment Log")
        print("="*50)
        print(f"{'Run':<5} {'Time(s)':<12} {'Nodes':<12} {'Cost'}")
        print("-"*50)
        for i, s in enumerate(self.stats_log, 1):
            print(f"{i:<5} {s['time_s']:<12} {s['nodes']:<12} {s['path_cost']}")
        n = len(self.stats_log)
        print("-"*50)
        print(f"{'Avg':<5} "
              f"{sum(x['time_s']    for x in self.stats_log)/n:<12.4f} "
              f"{sum(x['nodes']     for x in self.stats_log)/n:<12.1f} "
              f"{sum(x['path_cost'] for x in self.stats_log)/n:.1f}")
        print("="*50)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, 'example_map.txt')
    if not os.path.exists(map_path):
        print(f"[ERROR] Map file not found: {map_path}")
        sys.exit(1)
    Game(map_path).run()
