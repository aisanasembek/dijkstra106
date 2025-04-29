import pygame
import sys
import heapq
from collections import defaultdict
import time
import random

pygame.init()

WIDTH, HEIGHT = 1200, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dijkstra's Algorithm Visualizer")

FONT = pygame.font.SysFont('Arial', 16)
MEDIUM_FONT = pygame.font.SysFont('Arial', 18)
BIG_FONT = pygame.font.SysFont('Arial', 24)

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
BLUE = (66, 135, 245)
LIGHT_BLUE = (212, 230, 241)
DARK_BLUE = (20, 50, 90)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)
GREEN = (76, 187, 23)
LIGHT_GREEN = (200, 255, 200)
RED = (220, 0, 0)
LIGHT_RED = (255, 200, 200)
PURPLE = (142, 68, 173)
YELLOW = (241, 196, 15)
ORANGE = (230, 126, 34)
PINK = (255, 182, 193)

PRESET_GRAPHS = {
    "Simple Graph": {
        "graph": {
            0: {1: 4, 7: 8},
            1: {0: 4, 2: 8, 7: 11},
            2: {1: 8, 3: 7, 5: 4, 8: 2},
            3: {2: 7, 4: 9, 5: 14},
            4: {3: 9, 5: 10},
            5: {2: 4, 3: 14, 4: 10, 6: 2},
            6: {5: 2, 7: 1, 8: 6},
            7: {0: 8, 1: 11, 6: 1, 8: 7},
            8: {2: 2, 6: 6, 7: 7}
        },
        "positions": {
            0: (300, 330), 1: (450, 350), 2: (600, 300), 
            3: (750, 350), 4: (900, 400), 5: (750, 500), 
            6: (600, 550), 7: (450, 550), 8: (600, 400)
        }
    }
}

class GraphVisualizer:
    def __init__(self):
        self.start_node = 0
        self.reset_graph(PRESET_GRAPHS["Simple Graph"])
        
        self.steps = []
        self.step_index = 0
        self.animation_speed = 1
        self.playing = False
        self.visited = set()
        self.visited_edges = set()
        self.input_mode = None
        self.input_text = ""
        self.message = ""
        self.message_timer = 0
        self.representation_mode = "visual"
        self.show_table = True
        self.show_instuction = True
        self.show_legend = True

        self.edit_mode = False
        self.dragging_node = None
        self.creating_edge = None
        self.selected_node = None
        self.selected_edge = None
        self.last_click_time = 0
        self.last_click_pos = (0, 0)

        self.buttons = {
            "start": pygame.Rect(20, 750, 80, 30),
            "pause": pygame.Rect(110, 750, 80, 30),
            "previous": pygame.Rect(200, 750, 80, 30),  
            "next": pygame.Rect(290, 750, 80, 30),    
            "reset": pygame.Rect(380, 750, 80, 30),    
            "speed_down": pygame.Rect(470, 750, 30, 30),
            "speed_up": pygame.Rect(510, 750, 30, 30),
            "representation": pygame.Rect(560, 750, 100, 30),
            "table_toggle": pygame.Rect(670, 750, 100, 30),
            "legend_toggle": pygame.Rect(780, 750, 100, 30),
            "change_start": pygame.Rect(890, 750, 100, 30),
            "edit_toggle": pygame.Rect(1000, 750, 100, 30),
        }

        self.directed = False
        self.graph_type = "Preset"

        self.buttons["graph_type"] = pygame.Rect(900, 700, 130, 25)
        self.buttons["directed_toggle"] = pygame.Rect(1040, 700, 130, 25)


        
        self.preset_buttons = []
        self.update_steps()

    def reset_graph(self, graph_data):
        self.graph = defaultdict(dict)
        for node, edges in graph_data["graph"].items():
            self.graph[node] = edges.copy()
        self.positions = graph_data["positions"].copy()

        if self.positions:
            self.next_node_id = max(self.positions.keys()) + 1
        else:
            self.next_node_id = 0

        self.update_steps()

    def update_steps(self):
        try:
            if not self.graph:
                self.steps = []
                self.step_index = 0
                self.playing = False
                self.visited = set()
                self.visited_edges = set()
                return
            
            for node in self.positions:
                if node not in self.graph:
                    self.graph[node] = {}

            self.steps = self.dijkstra_steps(self.graph, self.start_node)
            self.step_index = 0
            self.playing = False
            self.visited = set()
            self.visited_edges = set()
            if self.steps:
                self.update_visited_sets()
        except Exception as e:
            self.steps = []
            self.playing = False
            self.message = f"Graph error: {str(e)}"
            self.message_timer = 180

    def dijkstra_steps(self, graph, start):
        if start not in graph:
            return []

        dist = {node: float('inf') for node in graph}
        dist[start] = 0
        prev = {node: None for node in graph} 
        visited = set()
        steps = [(start, dict(dist), set(visited.copy()), set())]
        pq = [(0, start)]
        visited_edges = set()

        while pq:
            current_dist, node = heapq.heappop(pq)
            if node in visited:
                continue

            edge = None
            if prev[node] is not None:
                edge = (prev[node], node)
                visited_edges.add(edge)

            visited.add(node)
            steps.append((node, dict(dist), set(visited.copy()), set(visited_edges.copy())))

            for neighbor, weight in graph[node].items():
                if dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight
                    prev[neighbor] = node
                    heapq.heappush(pq, (dist[neighbor], neighbor))

        self.prev = prev 
        return steps

    def get_shortest_path_edges(self, end):
        path_edges = set()
        node = end
        while self.prev.get(node) is not None:
            path_edges.add((self.prev[node], node))
            node = self.prev[node]
        return path_edges


    def add_node(self, pos):
        node_id = self.next_node_id
        self.graph[node_id] = {}
        self.positions[node_id] = pos
        self.next_node_id += 1
        self.update_steps()
        return node_id

    def add_edge(self, node1, node2, weight=1):
        if node1 == node2 or node2 in self.graph[node1]:
            return False

        self.graph[node1][node2] = weight
        
        if node2 not in self.graph:
            self.graph[node2] = {}
        
        if not self.directed:
            self.graph[node2][node1] = weight
        self.update_steps()
        return True


    def remove_node(self, node):
        self.next_node_id -= 1
        if node not in self.graph:
            return
        
        for neighbor in list(self.graph[node].keys()):
            del self.graph[neighbor][node]
        
        del self.graph[node]
        del self.positions[node]

        if self.start_node == node and self.graph:
            self.start_node = next(iter(self.graph.keys()))
        elif not self.graph:
            self.start_node = 0

        self.update_steps()

    def remove_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1]:
            del self.graph[node1][node2]
            del self.graph[node2][node1]
            self.update_steps()
            return True
        return False

    def update_edge_weight(self, node1, node2, weight):
        if node1 in self.graph and node2 in self.graph[node1]:
            self.graph[node1][node2] = weight
            if not self.directed and node2 in self.graph and node1 in self.graph[node2]:
                self.graph[node2][node1] = weight
            self.update_steps()
            return True
        return False

    
    def draw_instructions_table(self):
        if not self.edit_mode:
            return
            
        instructions = [
            "Left-click empty space: Add node",
            "Left-click node and drag: Move node",
            "Right-click node and drag: Create edge",
            "Double-click edge weight: Edit weight"
        ]

        x_start, y_start = 55, 70
        col_widths = [400]
        row_height = 30

        table_width = col_widths[0] + 10
        table_height = len(instructions) * row_height + 10
        pygame.draw.rect(
            SCREEN, LIGHT_BLUE,
            (x_start - 5, y_start - 5, table_width, table_height),
            border_radius=12
        )
        pygame.draw.rect(
            SCREEN, BLUE,
            (x_start - 5, y_start - 5, table_width, table_height),
            2, border_radius=12
        )

        for idx, instruction in enumerate(instructions):
            y = y_start + idx * row_height
            pygame.draw.rect(
                SCREEN, WHITE,
                (x_start, y, table_width - 10, row_height),
                border_radius=12
            )
            pygame.draw.rect(
                SCREEN, GRAY,
                (x_start, y, table_width - 10, row_height),
                1, border_radius=12
            )
            label = FONT.render(instruction, True, BLACK)
            SCREEN.blit(label, (x_start + 5, y + 5))

    def draw_dijkstra_table(self):
        if not self.steps or self.step_index >= len(self.steps):
            return
            
        header = ['Vertex', 'Visited', 'Distance', 'Path']
        x_start, y_start = 885, 70
        col_widths = [60, 60, 80, 60]
        row_height = 25

        table_width = sum(col_widths) + 10
        table_height = (len(self.graph) + 1) * row_height + 10
        pygame.draw.rect(
            SCREEN, LIGHT_BLUE,
            (x_start - 5, y_start - 5, table_width, table_height),
            border_radius=12
        )
        pygame.draw.rect(
            SCREEN, BLUE,
            (x_start - 5, y_start - 5, table_width, table_height),
            2, border_radius=12
        )

        for i, col in enumerate(header):
            x = x_start + sum(col_widths[:i])
            pygame.draw.rect(
                SCREEN, BLUE,
                (x, y_start, col_widths[i], row_height),
                border_radius=12
            )
            label = MEDIUM_FONT.render(col, True, WHITE)
            SCREEN.blit(label, (x + 5, y_start + 3))

        _, dist_snapshot, visited_nodes, _ = self.steps[self.step_index]
        for idx, node in enumerate(sorted(self.graph.keys())):
            y = y_start + (idx + 1) * row_height
            row_color = LIGHT_GREEN if node in visited_nodes else WHITE
            pygame.draw.rect(
                SCREEN, row_color,
                (x_start, y, table_width - 10, row_height),
                border_radius=12
            )
            pygame.draw.rect(
                SCREEN, GRAY,
                (x_start, y, col_widths[0], row_height),
                border_radius=12
            )
            label = FONT.render(str(node), True, BLUE if node == self.start_node else BLACK)
            SCREEN.blit(label, (x_start + 5, y + 5))

            visited = 'Yes' if node in visited_nodes else 'No'
            label = FONT.render(visited, True, GREEN if visited == 'Yes' else RED)
            SCREEN.blit(label, (x_start + col_widths[0] + 5, y + 5))

            cost = dist_snapshot[node]
            cost_display = "∞" if cost == float('inf') else str(cost)
            label = FONT.render(cost_display, True, BLACK)
            SCREEN.blit(label, (x_start + col_widths[0] + col_widths[1] + 5, y + 5))

            prev = -1
            for prev_node in self.graph:
                if (node in self.graph[prev_node] and
                    dist_snapshot[prev_node] + self.graph[prev_node][node] == dist_snapshot[node]):
                    prev = prev_node
                    break
            path_display = str(prev) if prev != -1 else "-"
            label = FONT.render(path_display, True, BLACK)
            SCREEN.blit(
                label,
                (x_start + col_widths[0] + col_widths[1] + col_widths[2] + 5, y + 5)
            )

    def draw_graph(self):
        shortest_path_edges = set()
        if self.steps and self.step_index < len(self.steps):
            current_node = self.steps[self.step_index][0]
            shortest_path_edges = self.get_shortest_path_edges(current_node)

        for node in self.graph:
            for neighbor, weight in self.graph[node].items():
                if not self.directed:
                    if node < neighbor:
                        start_pos = self.positions[node]
                        end_pos = self.positions[neighbor]

                        edge = (node, neighbor)
                        reversed_edge = (neighbor, node)

                        if edge in shortest_path_edges or reversed_edge in shortest_path_edges:
                            color = RED
                            width = 3
                        else:
                            color = GRAY
                            width = 2

                        pygame.draw.aaline(SCREEN, color, start_pos, end_pos, width)

                        mid_x = (start_pos[0] + end_pos[0]) // 2
                        mid_y = (start_pos[1] + end_pos[1]) // 2
                        weight_text = FONT.render(str(weight), True, DARK_BLUE)
                        SCREEN.blit(weight_text, (mid_x - 10, mid_y - 10))

                else:
                    start_pos = self.positions[node]
                    end_pos = self.positions[neighbor]

                    edge = (node, neighbor)

                    if edge in shortest_path_edges:
                        color = RED
                        width = 3
                    else:
                        color = GRAY
                        width = 2

                    pygame.draw.aaline(SCREEN, color, start_pos, end_pos, width)

                    self.draw_arrowhead(start_pos, end_pos, color)

                    mid_x = int(start_pos[0] * 0.7 + end_pos[0] * 0.3)
                    mid_y = int(start_pos[1] * 0.7 + end_pos[1] * 0.3)
                    weight_text = FONT.render(str(weight), True, DARK_BLUE)
                    SCREEN.blit(weight_text, (mid_x - 10, mid_y - 10))


        current_node = None
        if self.steps and self.step_index < len(self.steps):
            current_node = self.steps[self.step_index][0]

        visited_nodes = set()
        if self.steps and self.step_index < len(self.steps):
            _, _, visited_nodes, _ = self.steps[self.step_index]

        for node, pos in self.positions.items():
            if node == current_node:
                color = ORANGE
            elif node in visited_nodes:
                color = GREEN
            elif node == self.start_node:
                color = ORANGE
            elif self.edit_mode and node == self.selected_node:
                color = PINK
            else:
                color = BLUE

            shadow_pos = (pos[0] + 3, pos[1] + 3)
            pygame.draw.circle(SCREEN, GRAY, shadow_pos, 22)

            pygame.draw.circle(SCREEN, color, pos, 20)
            pygame.draw.circle(SCREEN, BLACK, pos, 20, 1)

            text = MEDIUM_FONT.render(str(node), True,
                                    WHITE if color not in (YELLOW, PINK) else BLACK)
            text_rect = text.get_rect(center=pos)
            SCREEN.blit(text, text_rect)

        if self.creating_edge and self.creating_edge in self.positions:
            start_pos = self.positions[self.creating_edge]
            end_pos = pygame.mouse.get_pos()
            pygame.draw.aaline(SCREEN, PINK, start_pos, end_pos)


    def draw_arrowhead(self, start, end, color):
        if start == end:
            return

        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return

        ux, uy = dx / length, dy / length

        node_radius = 20
        shrink = node_radius + 5 

        tip_x = end[0] - ux * shrink
        tip_y = end[1] - uy * shrink

        base_x = tip_x - ux * 10  
        base_y = tip_y - uy * 10

        left_x = base_x - uy * 7
        left_y = base_y + ux * 7

        right_x = base_x + uy * 7
        right_y = base_y - ux * 7

        pygame.draw.polygon(SCREEN, color, [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)])


    def draw_adjacency_list(self):
        x_start, y_start = 50, 100
        row_height = 20

        title = BIG_FONT.render("Adjacency List Representation:", True, BLUE)
        SCREEN.blit(title, (x_start, y_start - 30))

        list_height = len(self.graph) * row_height + 10
        pygame.draw.rect(SCREEN, LIGHT_BLUE,
                         (x_start - 5, y_start - 5, 350, list_height),
                         border_radius=12)
        pygame.draw.rect(SCREEN, BLUE,
                         (x_start - 5, y_start - 5, 350, list_height),
                         2, border_radius=12)

        current_node = None
        if self.steps and self.step_index < len(self.steps):
            current_node = self.steps[self.step_index][0]

        for idx, node in enumerate(sorted(self.graph.keys())):
            neighbors = sorted(self.graph[node].items())
            neighbor_text = ", ".join([f"{n}({w})" for n, w in neighbors]) if neighbors else "None"
            
            if node == current_node:
                pygame.draw.rect(SCREEN, LIGHT_GREEN,
                                 (x_start, y_start + idx * row_height, 340, row_height),
                                 border_radius=12)

            node_label = FONT.render(f"{node}:", True, BLUE if node == self.start_node else BLACK)
            neighbor_label = FONT.render(neighbor_text, True, BLACK)

            SCREEN.blit(node_label, (x_start, y_start + idx * row_height))
            SCREEN.blit(neighbor_label, (x_start + 35, y_start + idx * row_height))

    def draw_controls(self):
        pygame.draw.rect(SCREEN, LIGHT_BLUE, (0, 730, WIDTH, 70), border_radius=12)
        pygame.draw.rect(SCREEN, BLUE, (0, 730, WIDTH, 70), 2, border_radius=12)

        draw_button(self.buttons["start"], "Play", not self.playing, GREEN)
        draw_button(self.buttons["pause"], "Pause", self.playing, RED)
        draw_button(self.buttons["previous"], "Previous", self.step_index > 0, BLUE)
        draw_button(self.buttons["next"], "Next", self.step_index < len(self.steps) - 1, BLUE)
        draw_button(self.buttons["reset"], "Reset", True, DARK_GRAY)

        draw_button(self.buttons["speed_down"], "-", self.animation_speed > 0.5, BLUE)
        draw_button(self.buttons["speed_up"], "+", self.animation_speed < 5, BLUE)
        speed_text = FONT.render(f"Speed: {self.animation_speed}", True, BLACK)
        SCREEN.blit(speed_text, (475, 730))

        draw_button(self.buttons["representation"],
                    "Visual" if self.representation_mode == "visual" else "Adjacency",
                    True, PURPLE)
        draw_button(self.buttons["table_toggle"],
                    "Hide Table" if self.show_table else "Show Table",
                    True, PURPLE)
        draw_button(self.buttons["legend_toggle"],
                    "Hide Legend" if self.show_legend else "Show Legend",
                    True, PURPLE)
        
        draw_button(self.buttons["change_start"], "Change Start", True, DARK_GRAY)
        draw_button(self.buttons["edit_toggle"],
                    "Exit Edit" if self.edit_mode else "Edit Graph",
                    True, PINK if self.edit_mode else DARK_GRAY)
        draw_button(self.buttons["graph_type"], f"Graph: {self.graph_type}", True, DARK_GRAY)
        draw_button(self.buttons["directed_toggle"],
                    "Directed" if self.directed else "Undirected",
                    True, DARK_GRAY)


        step_text = FONT.render(f"Step: {self.step_index}/{len(self.steps)-1 if self.steps else 0}", True, BLACK)
        SCREEN.blit(step_text, (20, 730))

    def draw_legend(self):
        if not self.show_legend:
            return

        legend_x, legend_y = 50, 10
        legend_width = 1100
        legend_height = 50

        pygame.draw.rect(SCREEN, LIGHT_BLUE,
                        (legend_x, legend_y, legend_width, legend_height),
                        border_radius=15)
        pygame.draw.rect(SCREEN, BLUE,
                        (legend_x, legend_y, legend_width, legend_height),
                        2, border_radius=15)

        title = FONT.render("Legend:", True, BLUE)
        SCREEN.blit(title, (legend_x + 10, legend_y + 15))

        items = [
            (ORANGE, "Current"),
            (GREEN, "Visited"),
            (BLUE, "Unvisited"),
            (GRAY, "Edge"),
            (RED, "Shortest Edge"),
            (PINK, "Selected/Creating")
        ]
        
        x_offset = legend_x + 100
        spacing = 140  
        for i, (color, text) in enumerate(items):
            rect_x = x_offset + i * spacing
            rect_y = legend_y + 10
            
            if color in (GRAY, RED):  
                pygame.draw.line(SCREEN, color, (rect_x - 5, rect_y + 10), (rect_x + 15, rect_y + 10), 3)
            else:
                pygame.draw.circle(SCREEN, color, (rect_x, rect_y + 10), 8)

            label = FONT.render(text, True, BLACK)
            SCREEN.blit(label, (rect_x + 20, rect_y + 2))

    def draw_message(self):
        if self.message and self.message_timer > 0:
            msg_bg = pygame.Rect(20, 700, 400, 25)
            pygame.draw.rect(SCREEN, LIGHT_RED, msg_bg, border_radius=12)
            pygame.draw.rect(SCREEN, RED, msg_bg, 2, border_radius=12)
            msg_surface = FONT.render(self.message, True, RED)
            SCREEN.blit(msg_surface, (msg_bg.x + 10, msg_bg.y + 3))
            self.message_timer -= 1
        else:
            self.message = ""

    def handle_click(self, pos):
        mx, my = pos

        if my < 730:
            if self.edit_mode:
                self.handle_edit_click(pos)
                return
            elif self.input_mode == "change_start":
                self.handle_start_node_selection(pos)
                return

        for name, button in self.buttons.items():
            if button.collidepoint(mx, my):
                if name == "start":
                    self.playing = True
                elif name == "pause":
                    self.playing = False
                elif name == "step":
                    if self.step_index < len(self.steps) - 1:
                        self.step_index += 1
                        self.update_visited_sets()
                elif name == "previous":
                    if self.step_index > 0:
                        self.step_index -= 1
                        self.update_visited_sets()
                elif name == "next":
                    if self.step_index < len(self.steps) - 1:
                        self.step_index += 1
                        self.update_visited_sets()
                elif name == "reset":
                    self.reset_graph(PRESET_GRAPHS["Simple Graph"])
                    self.message = "Graph reset to Simple Graph"
                    self.message_timer = 120
                elif name == "speed_down" and self.animation_speed > 0.5:
                    self.animation_speed -= 0.5
                elif name == "speed_up" and self.animation_speed < 5:
                    self.animation_speed += 0.5
                elif name == "representation":
                    self.representation_mode = "adjacency" if self.representation_mode == "visual" else "visual"
                elif name == "table_toggle":
                    self.show_table = not self.show_table
                elif name == "legend_toggle":
                    self.show_legend = not self.show_legend
                elif name == "change_start":
                    if not self.graph:
                        self.message = "Graph is empty!"
                        self.message_timer = 120
                    else:
                        self.input_mode = "change_start"
                        self.message = "Click on a node to set as start"
                        self.message_timer = 120
                elif name == "edit_toggle":
                    self.edit_mode = not self.edit_mode
                    self.selected_node = None
                    self.creating_edge = None
                    if self.edit_mode:
                        self.playing = False
                        self.message = "Edit mode activated"
                    else:
                        self.message = "Edit mode deactivated"
                    self.message_timer = 120
                elif name == "graph_type":
                    options = ["Preset", "Empty", "Random"]
                    index = (options.index(self.graph_type) + 1) % len(options)
                    self.graph_type = options[index]
                    self.load_graph_type()
                    self.message = f"Graph type: {self.graph_type}"
                    self.message_timer = 120

                elif name == "directed_toggle":
                    self.directed = not self.directed
                    
                    if not self.directed:
                        new_graph = defaultdict(dict)
                        for node in self.graph:
                            for neighbor, weight in self.graph[node].items():
                                if neighbor not in new_graph or node not in new_graph[neighbor]:
                                    new_graph[node][neighbor] = weight
                        self.graph = new_graph

                    mode = "Directed" if self.directed else "Undirected"
                    self.message = f"Edges are now {mode}"
                    self.message_timer = 120


                return

    def handle_delete_key(self):
        if not self.edit_mode:
            return
            
        if self.selected_node is not None:
            if self.selected_node == self.start_node:
                self.message = "Cannot delete start node!"
            else:
                self.remove_node(self.selected_node)
                self.message = f"Removed node {self.selected_node}"
                self.selected_node = None
        elif self.selected_edge is not None:
            node1, node2 = self.selected_edge
            self.remove_edge(node1, node2)
            self.message = f"Removed edge {node1}-{node2}"
            self.selected_edge = None
        else:
            self.message = "Nothing selected to delete"
        self.message_timer = 120

    def handle_edit_click(self, pos):
        if not self.edit_mode:
            return
            
        current_time = time.time()
        if pygame.mouse.get_pressed()[0]:
            for node in list(self.graph.keys()):
                for neighbor in list(self.graph[node].keys()):
                    if not self.directed or (self.directed and (node, neighbor) not in [(n, node) for n in self.graph[neighbor]]):
                        start_pos = self.positions[node]
                        end_pos = self.positions[neighbor]
                        mid_x = (start_pos[0] + end_pos[0]) // 2
                        mid_y = (start_pos[1] + end_pos[1]) // 2
                        if abs(mid_x - pos[0]) < 20 and abs(mid_y - pos[1]) < 10:
                            if (current_time - self.last_click_time < 0.4 and
                                abs(mid_x - self.last_click_pos[0]) < 10 and
                                abs(mid_y - self.last_click_pos[1]) < 10):
                                self.edit_edge_weight(node, neighbor)
                            else:
                                self.selected_edge = (node, neighbor)
                                self.selected_node = None
                                self.last_click_time = current_time
                                self.last_click_pos = (mid_x, mid_y)
                            return

        clicked_node = None
        for node, (x, y) in self.positions.items():
            if (x - pos[0])**2 + (y - pos[1])**2 <= 400:
                clicked_node = node
                break

        if pygame.mouse.get_pressed()[0]:
            if clicked_node is not None:
                self.dragging_node = clicked_node
                self.selected_node = clicked_node
                self.selected_edge = None
            else:
                for node in self.graph:
                    for neighbor in self.graph[node]:
                        if node < neighbor:
                            if self.point_near_line(pos, self.positions[node], self.positions[neighbor]):
                                self.selected_edge = (node, neighbor)
                                self.selected_node = None
                                return
                
                node_id = self.add_node(pos)
                self.message = f"Added node {node_id}"
                self.message_timer = 120
                self.selected_node = node_id
                self.selected_edge = None

        elif pygame.mouse.get_pressed()[2]:
            if clicked_node is not None:
                if self.creating_edge is None:
                    self.creating_edge = clicked_node
                    self.selected_node = clicked_node
                else:
                    if self.creating_edge != clicked_node:
                        self.add_edge(self.creating_edge, clicked_node)
                        self.message = f"Added edge {self.creating_edge}-{clicked_node}"
                        self.message_timer = 120
                    self.creating_edge = None
                    self.selected_node = None

    def handle_start_node_selection(self, pos):
        for node, (x, y) in self.positions.items():
            if (x - pos[0])**2 + (y - pos[1])**2 <= 400:
                self.start_node = node
                self.update_steps()
                self.input_mode = None
                self.message = f"Start node set to {node}"
                self.message_timer = 120
                return
        self.message = "Click on a node to set as start"
        self.message_timer = 120

    def edit_edge_weight(self, node1, node2):
        input_active = True
        input_text = str(self.graph[node1][node2])
        input_rect = pygame.Rect(400, 300, 200, 30)

        while input_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        try:
                            weight = int(input_text)
                            if weight > 0:
                                self.update_edge_weight(node1, node2, weight)
                            input_active = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_ESCAPE:
                        input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isdigit():
                        input_text += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not input_rect.collidepoint(event.pos):
                        input_active = False

            pygame.draw.rect(SCREEN, WHITE, (390, 280, 220, 80), border_radius=12)
            pygame.draw.rect(SCREEN, BLUE, (390, 280, 220, 80), 2, border_radius=12)

            prompt = FONT.render(f"Edge {node1}-{node2} weight:", True, BLACK)
            SCREEN.blit(prompt, (400, 280))

            pygame.draw.rect(SCREEN, LIGHT_BLUE, input_rect, border_radius=12)
            pygame.draw.rect(SCREEN, BLUE, input_rect, 2, border_radius=12)

            text_surface = FONT.render(input_text, True, BLACK)
            SCREEN.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

            pygame.display.flip()

    def point_near_line(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_len = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if line_len == 0:
            return False
        distance = abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1) / line_len
        return (distance < 5 and
                min(x1, x2) - 5 <= x <= max(x1, x2) + 5 and
                min(y1, y2) - 5 <= y <= max(y1, y2) + 5)

    def update_visited_sets(self):
        if self.steps and self.step_index < len(self.steps):
            _, _, self.visited, self.visited_edges = self.steps[self.step_index]

    def load_graph_type(self):
        if self.graph_type == "Preset":
            self.reset_graph(PRESET_GRAPHS["Simple Graph"])
        elif self.graph_type == "Empty":
            self.reset_graph({"graph": {}, "positions": {}})
        elif self.graph_type == "Random":
            self.generate_random_graph()

    def generate_random_graph(self, node_count=8):
        self.graph = defaultdict(dict)
        self.positions = {}
        width_margin, height_margin = 150, 200

        for i in range(node_count):
            x = random.randint(width_margin, WIDTH - width_margin)
            y = random.randint(height_margin, HEIGHT - height_margin)
            self.positions[i] = (x, y)
            self.graph[i] = {} 

        self.next_node_id = node_count

        for i in range(node_count):
            for j in range(i + 1, node_count):
                if random.random() < 0.35:
                    weight = random.randint(1, 10)
                    self.graph[i][j] = weight
                    if not self.directed:
                        self.graph[j][i] = weight

        self.update_steps()


    def update(self):
        if self.playing and self.step_index < len(self.steps) - 1:
            frames_per_step = max(1, 30 - (self.animation_speed * 5))
            if pygame.time.get_ticks() % frames_per_step == 0:
                self.step_index += 1
                self.update_visited_sets()
            if self.step_index >= len(self.steps) - 1:
                self.playing = False

        if self.edit_mode and self.dragging_node is not None:
            if pygame.mouse.get_pressed()[0]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.positions[self.dragging_node] = (mouse_x, mouse_y)
            else:
                self.dragging_node = None

    def draw(self):
        draw_background_gradient()
        self.draw_legend()

        if self.representation_mode == "visual":
            self.draw_graph()
        else:
            self.draw_graph()
            self.draw_adjacency_list()

        if self.show_table:
            self.draw_dijkstra_table()

        self.draw_instructions_table()
        self.draw_controls()
        self.draw_message()

def draw_button(rect, text, active=True, color=BLUE):
    shadow_rect = rect.copy()
    shadow_rect.x += 3
    shadow_rect.y += 3
    pygame.draw.rect(SCREEN, GRAY, shadow_rect, border_radius=12)
    pygame.draw.rect(SCREEN, color if active else GRAY, rect, border_radius=12)
    pygame.draw.rect(SCREEN, BLACK, rect, 1, border_radius=12)
    label = FONT.render(text, True, WHITE if color != LIGHT_BLUE else BLACK)
    label_rect = label.get_rect(center=rect.center)
    SCREEN.blit(label, label_rect)

def draw_background_gradient():
    top_color = WHITE
    bottom_color = LIGHT_BLUE
    for y in range(HEIGHT):
        blend = y / HEIGHT
        r = int(top_color[0] * (1 - blend) + bottom_color[0] * blend)
        g = int(top_color[1] * (1 - blend) + bottom_color[1] * blend)
        b = int(top_color[2] * (1 - blend) + bottom_color[2] * blend)
        pygame.draw.line(SCREEN, (r, g, b), (0, y), (WIDTH, y))

def main():
    clock = pygame.time.Clock()
    visualizer = GraphVisualizer()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                visualizer.handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    visualizer.playing = not visualizer.playing
                elif event.key == pygame.K_RIGHT and visualizer.step_index < len(visualizer.steps) - 1:
                    visualizer.step_index += 1
                    visualizer.update_visited_sets()
                elif event.key == pygame.K_LEFT and visualizer.step_index > 0:
                    visualizer.step_index -= 1
                    visualizer.update_visited_sets()
                elif event.key == pygame.K_DELETE:
                    visualizer.handle_delete_key()
                elif event.key == pygame.K_RIGHT and visualizer.step_index < len(visualizer.steps) - 1:
                    visualizer.step_index += 1
                    visualizer.update_visited_sets()
                elif event.key == pygame.K_LEFT and visualizer.step_index > 0:
                    visualizer.step_index -= 1
                    visualizer.update_visited_sets()

        visualizer.update()
        visualizer.draw()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()