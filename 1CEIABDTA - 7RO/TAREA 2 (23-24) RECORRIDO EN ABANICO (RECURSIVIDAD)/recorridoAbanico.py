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
MAZE[3,1] =1
MARK = MAZE[:]
                  #[y, x]
MOVE = np.array([ (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1) ])  # arriba, arriba-derecha, derecha, abajo-derecha, abajo, abajo-izquierda, izquierda, arriba-izquierda

def neighbors(x, y):
    result = { (int(y+ dy), int(x + dx)): MARK[int(y+ dy), int(x + dx)] for dy, dx in MOVE}
    return dict(result)

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

class Place: 
    def __init__(self, fila, columna):
        self.fila = fila
        self.columna = columna
    def mostrar(self):
        print("( " + self.fila+ "," +self.columna+")")

walker = Agente(1, 1, 2, 3)  #Este recorrerá el laberinto en busca de la meta
goal =   Agente(8, 8, 4)  #Este será la meta

def maze2graph(maze):
    graph = {}
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] != 1:  # Assuming 1 represents walls
                graph[(y, x)] = []
                if y > 0 and maze[y-1][x] != 1:
                    graph[(y, x)].append(("U", (y-1, x)))  # Down
                if x < len(maze[0]) - 1 and maze[y][x+1] != 1:
                    graph[(y, x)].append(("R", (y, x+1)))  # Right
                if y < len(maze) - 1 and maze[y+1][x] != 1:
                    graph[(y, x)].append(("D", (y+1, x)))  # Up
                if x > 0 and maze[y][x-1] != 1:
                    graph[(y, x)].append(("L", (y, x-1)))  # Left
    return graph

def find_path_dfs(maze):
    start, goal = (1, 1), (len(maze) - 2, len(maze[0]) - 2)
    stack = [("", start)]
    visited = set()
    graph = maze2graph(maze)
    
    while stack:
        path, current = stack.pop()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            stack.append((path + direction, neighbour))
    
    return "NO WAY!"

def visualize_example(x): 
    plt.figure()
    plt.imshow(x)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Encontrar el camino
path = find_path_dfs(MARK)
print("Path found:", path)
equivalences = {"R":(0,1), "D": (1,0), "L":(0,-1), "U":(-1,0)}
for direction in path:
    walker.mover(equivalences[direction], MARK)


visualize_example(MARK)