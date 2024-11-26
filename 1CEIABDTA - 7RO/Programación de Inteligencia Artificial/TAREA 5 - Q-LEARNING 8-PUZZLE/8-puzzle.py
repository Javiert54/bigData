import numpy as np
import random

class QLearningAgent:
    def __init__(self, table, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.table = table
        self.originalTable = table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((len(table.MAZE), 4))  # 4 posibles acciones: arriba, abajo, izquierda, derecha

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Acción aleatoria
        else:
            return np.argmax(self.q_table[state])  # Mejor acción según la Q-table

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, episodes=1000):
        for _ in range(episodes):
            state = self.originalTable
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                self.learn(state, action, reward, next_state)
                state = next_state


    def step(self, state, action):
        # Implementa la lógica para ejecutar una acción y devolver el siguiente estado, la recompensa y si el episodio ha terminado
        pass


class table:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.MAZE = []
        self.MAZE = [0] * (self.y * self.x)
        # self.MAZE[6] = 1
        # self.MAZE[8] = 1
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
                
tabla = table(3, 3) #Enseña el mazo
tabla.__str__()
print(preCondiciones(tabla, 2, 5))
# Ejemplo de uso
tabla = table(3, 3)
agent = QLearningAgent(tabla)
agent.train()
