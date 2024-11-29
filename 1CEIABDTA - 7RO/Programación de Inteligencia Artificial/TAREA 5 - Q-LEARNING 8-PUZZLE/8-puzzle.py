import numpy as np
import random

directions = [-3, 1, 3, -1]  # Change in index for each move
move_names = ["Up", "Right", "Down", "Left"]

# Validate if a move is within bounds
def is_valid_move(index, direction):
    if direction == -3 and index < 3:  # Up
        return False
    if direction == 3 and index > 5:  # Down
        return False
    if direction == -1 and index % 3 == 0:  # Left
        return False
    if direction == 1 and index % 3 == 2:  # Right
        return False
    return True


class table:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.MAZE = [str(i+1) for i in range((self.y * self.x)-1)]
        self.MAZE.append(" ")
    def __str__(self):
        aux = 0
        for i in range(self.x, (self.y+1)*self.x, self.x):
            print(self.MAZE[aux:i])
            aux = i


def preCondiciones(table:table, agentPosition:int, destinationPosition:int):
    PC = [None]*(len(table.MAZE)) #PreCondiciones

    agentColumn:[int] = agentPosition % table.x
    agentRow:[int] = (agentPosition - agentColumn) // table.x
    
    destinationColumn:[int] = destinationPosition % table.x
    destinationRow:[int] = (destinationPosition - destinationColumn) // table.x
    try:
        for i in range(agentColumn, agentPosition, table.x): #Podemos iterar entre las celdas, saltando el mismo número de filas que table.x para aterrizar en la misma columna
            if PC[i]!= None: #Si ya se ha definido una precondición para la posición
                return None #No se puede llegar a la posición
            PC[i] = False #No debe haber huecos


        if agentRow < (len(table.MAZE)/table.x)-1: #Si el agente no está en la base
            for i in range(agentPosition, len(table.MAZE)+1, table.x): 
                if PC[i]!= None: #Si ya se ha definido una precondición para la posición
                    return None #No se puede llegar a la posición
                PC[i] = True #Debe haber bloques por debajo del agente
        PC[agentPosition] = True #Debe haber bloque donde está el agente


        for i in range(destinationColumn, destinationPosition, table.x): 
            if PC[i]!= None: #Si ya se ha definido una precondición para la posición
                return None #No se puede llegar a la posición
            PC[i] = False #Debe haber hueco por encima del destino


        if destinationRow < (len(table.MAZE)/table.x)-1:
            for i in range(destinationPosition, len(table.MAZE), table.x): 
                if PC[i]!= None: #Si ya se ha definido una precondición para la posición
                    return None #No se puede llegar a la posición
                PC[i] = True #Debe haber bloques por debajo del destino
        PC[destinationPosition] = False #Debe haber hueco donde está el destino

        return PC  #Se devuelve la lista de PreCondiciones
    except:
        return None  #Si ocurre algún error, devolvemos None
                
# Swap two positions in a state
def swap(state, i, j):
    state = list(state)
    state[i], state[j] = state[j], state[i]
    return ''.join(state)

# Generate transitions for the 8-puzzle
def generate_transitions_8_puzzle(initial_state):   #  POR REVISAR !!!!
    visited = set()
    queue = deque([initial_state])
    transitions = {}

    while queue:
        current_state = queue.popleft()
        if current_state in visited:
            continue
        visited.add(current_state)

        empty_index = current_state.index('9')
        state_transitions = []

        for direction in directions:
            if is_valid_move(empty_index, direction):
                new_state = swap(current_state, empty_index, empty_index + direction)
                state_transitions.append(new_state)
                if new_state not in visited:
                    queue.append(new_state)
            else:
                state_transitions.append(None)  # Invalid move

        transitions[current_state] = state_transitions

    return transitions

# Initial state of the puzzle
initial_state = "123456789"

# Generate the transitions
transitions = generate_transitions_8_puzzle(initial_state)

tabla = table(3, 3) #Enseña el mazo
tabla.__str__()
print(preCondiciones(tabla, 2, 5))
