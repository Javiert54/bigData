import numpy as np
import matplotlib.pyplot as plt

# Crear un array de 10x10 lleno de ceros
def createMaze(x, y):
    MAZE = np.zeros((x, y))
    MAZE[0, :] = 1
    MAZE[-1, :] = 1
    MAZE[:, 0] = 1
    MAZE[:, -1] = 1
    return MAZE

MAZE = createMaze(10, 10)
MAZE[1,3] =1  #Agregamos un obstáculos
MAZE[2,3] =1
MAZE[3,4] =1
MARK = MAZE[:]

class Agente:
    def __init__(self, x, y, color, traceColor=0):
        self.x = x
        self.y = y
        self.color = color
        self.trace = traceColor
        MARK[self.y, self.x] = self.color

    def mover(self, direccion, mark):
        print(mark)
        mark[self.y, self.x] = self.trace
        self.x += direccion[1]
        self.y += direccion[0]
        print("moviendo a: ", self.y, self.x)
        mark[self.y, self.x] = self.color #Esto es para que marque de color donde ya hemos estado

walker = Agente(1, 1, 2, 3)  #Este recorrerá el laberinto en busca de la meta
goal =   Agente(8, 8, 4)  #Este será la meta

def maze2graph(maze):
    graph = {}
    # iterate over the cells of the maze
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] != 1:  # If it is not a wall
                graph[(y, x)] = []  # Create an empty list for the neighbors of the current cell and the directions to reach them
                if maze[y-1][x] != 1:  
                    graph[(y, x)].append(((-1,0), (y-1, x)))  # Down
                if maze[y-1][x+1] != 1:
                    graph[(y, x)].append(((-1,1), (y-1, x+1)))  # Down-Right
                if maze[y][x+1] != 1:
                    graph[(y, x)].append(((0,1), (y, x+1)))  # Right
                if maze[y+1][x+1] != 1:
                    graph[(y, x)].append(((1,1), (y+1, x+1)))  # Up-Right
                if maze[y+1][x] != 1:
                    graph[(y, x)].append(((1,0), (y+1, x)))  # Up
                if maze[y+1][x-1] != 1:  
                    graph[(y, x)].append(((1,-1), (y+1, x-1)))  # Up-Left
                if maze[y][x-1] != 1:
                    graph[(y, x)].append(((0,-1), (y, x-1)))  # Left
                if maze[y-1][x-1] != 1:
                    graph[(y, x)].append(((-1,-1), (y-1, x-1)))  # Down-Left
    return graph
 
def find_path_dfs(graph, current, goal, path=(), visited=None):
    if visited is None:
        visited = set()
    if current == goal:
        return path  # Return the path if we have reached the goal
    if current in visited:
        return None
    visited.add(current)

    for direction, neighbour in graph[current]:  # Iterate over the neighbors of the current cell
        result = find_path_dfs(graph, neighbour, goal, path + (direction,), visited)  # Recursively search for the path, adding the direction to the path
        if result is not None:
            return result
    return "NO WAY!" # It will return this if there is no possible path to reach the goal

def visualize_example(maze): 
    plt.figure()
    plt.imshow(maze)
    plt.colorbar()
    plt.grid(False)
    plt.show()


start, goal = (1, 1), (len(MARK) - 2, len(MARK[0]) - 2)
path = find_path_dfs(maze2graph(MARK), start, goal)
print("Path found:", path)

for direction in path:
    walker.mover(direction, MARK)  # Move the walker according to the path

visualize_example(MARK)