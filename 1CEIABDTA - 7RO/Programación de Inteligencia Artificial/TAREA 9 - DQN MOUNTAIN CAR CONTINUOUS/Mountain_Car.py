import gym
import numpy as np
import matplotlib.pyplot as plt
import os # Necesario para crear el directorio de vídeos
from gym.wrappers import RecordVideo # Importar el wrapper para grabar vídeo

# --- Configuración ---
env_id = 'MountainCarContinuous-v0'
video_folder = 'videos_mountain_car_continuous' # Carpeta para guardar los vídeos
record_interval = 100 # Grabar un vídeo cada 100 episodios

# Crear el directorio de vídeos si no existe
if not os.path.exists(video_folder):
    os.makedirs(video_folder)
    print(f"Directorio creado: {video_folder}")

# Crear el entorno CON render_mode='rgb_array' para poder grabar
# NOTA IMPORTANTE: El algoritmo Q-learning con discretización de estados
# como está implementado aquí NO es adecuado para el espacio de acciones
# continuo de 'MountainCarContinuous-v0'. El agente tomará principalmente
# acciones aleatorias y no aprenderá una política efectiva con esta configuración.
# La grabación mostrará este comportamiento aleatorio.
env = gym.make(env_id, render_mode='rgb_array')

# --- Parámetros del Agente (Q-Learning Discretizado - Problemático aquí) ---
episodes = 1000
max_steps_per_episode = 200 # Límite de pasos por episodio
learning_rate = 0.01
discount_factor = 0.99 # Factor de descuento (debe ser < 1)

# Discretizar el espacio de estados
state_bins_count = [20, 20]  # Número de 'cajas' para cada dimensión del estado
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Ajustar ligeramente los límites para asegurar que los valores extremos caigan en los bins
# A veces es necesario por precisión flotante
state_bounds[0] = (state_bounds[0][0], state_bounds[0][1] + 1e-6) # Posición
state_bounds[1] = (state_bounds[1][0], state_bounds[1][1] + 1e-6) # Velocidad

# Calcular los bordes de los bins (excluyendo los extremos para np.digitize)
state_bins = [np.linspace(bounds[0], bounds[1], num=bins + 1)[1:-1]
              for bounds, bins in zip(state_bounds, state_bins_count)]

def discretize_state(state):
    """Discretiza un estado continuo en una tupla de índices discretos."""
    state_idx = []
    for i, value in enumerate(state):
        # Asegurar que el valor esté dentro de los límites antes de discretizar
        clipped_value = np.clip(value, env.observation_space.low[i], env.observation_space.high[i])
        state_idx.append(np.digitize(clipped_value, state_bins[i]))
    return tuple(state_idx)

# Inicializar Q-table (Estructura incorrecta para acciones continuas)
# El tamaño [1] para la acción es un placeholder y no representa el espacio continuo.
q_table_shape = [len(bins) + 1 for bins in state_bins] + [1]
q_table = np.zeros(q_table_shape)

# Función para elegir acción (Actualmente aleatoria)
def choose_action(state):
    # Idealmente, esto usaría la Q-table, pero la estructura actual no lo permite
    # para acciones continuas. Se usan acciones aleatorias.
    return env.action_space.sample()
# --- Fin Configuración Q-Learning ---


# --- Envolver el entorno para grabar vídeo ---
# Se grabarán los episodios cuyo número sea múltiplo de record_interval (0, 100, 200, ...)
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda e: e % record_interval == 0, # Condición para grabar
    name_prefix=f"{env_id}-qlearning-discrete" # Prefijo para los nombres de archivo
)
# --- Fin Envoltura Vídeo ---


# --- Bucle de Entrenamiento ---
rewards = []
print(f"Iniciando entrenamiento ({episodes} episodios)...")
print(f"Se grabarán vídeos cada {record_interval} episodios en '{video_folder}'")

for episode in range(episodes):
    # env.reset() devuelve state, info
    state_continuous, info = env.reset()
    state = discretize_state(state_continuous)
    total_reward = 0
    terminated = False
    truncated = False
    step_count = 0

    # Bucle hasta que el episodio termine (terminated o truncated)
    while not terminated and not truncated:
        # El wrapper RecordVideo llama a env.render() internamente si es necesario

        action = choose_action(state) # Acción (aleatoria en este caso)
        # Asegurar que la acción esté estrictamente dentro de los límites
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_state_continuous, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state_continuous)
        total_reward += reward

        # Actualización Q-table (conceptualemente errónea para acciones continuas)
        try:
            current_q = q_table[state][0] # Asume una única columna de acción
            next_max_q = np.max(q_table[next_state]) # Asume una única columna de acción
            new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
            q_table[state][0] = new_q
        except IndexError:
            # Esto puede ocurrir si la discretización genera índices fuera de rango
            # print(f"Advertencia: Índice fuera de rango. state={state}, next_state={next_state}, shape={q_table.shape}")
            pass # Ignorar actualización si hay error de índice

        state = next_state
        step_count += 1

        # Truncar si se excede el número máximo de pasos
        if step_count >= max_steps_per_episode:
            truncated = True

    rewards.append(total_reward)

    # Imprimir progreso cada 10 episodios o en el último
    if (episode + 1) % 10 == 0 or episode == episodes - 1:
        print(f"Episodio {episode + 1}/{episodes}: Recompensa Total: {total_reward:.2f}, Pasos: {step_count}")

    # El wrapper RecordVideo gestiona el guardado del vídeo cuando termina un episodio grabado
    # env.episode_id es gestionado por el wrapper RecordVideo
    if env.episode_id > 0 and env.episode_id % record_interval == 0:
         # El episodio_id empieza en 0, así que comprobamos después de que se complete
         if episode == env.episode_id -1 : # Asegurarse que el mensaje sale tras el episodio correcto
             print(f"-> Vídeo grabado para episodio {env.episode_id}")


# --- Finalización y Gráfica ---
print("Cerrando entorno y finalizando vídeos...")
env.close() # ¡Importante! Cierra el entorno y finaliza el último vídeo si estaba grabándose.
print(f"Entrenamiento completado. Vídeos guardados en: {video_folder}")

# Graficar las recompensas
plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Recompensa por Episodio', alpha=0.6)

# Media móvil para suavizar la curva
window_size = 50
if len(rewards) >= window_size:
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=f'Media Móvil ({window_size} episodios)', color='orange', linewidth=2)
else:
    print("No hay suficientes episodios para calcular la media móvil.")

plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.title('Recompensas Durante el Entrenamiento (Acciones Aleatorias)') # Título ajustado
plt.legend()
plt.grid(True)
plt.savefig('mountain_car_continuous_rewards_qlearning_discrete.png') # Guardar la gráfica
plt.show()

print("Script finalizado.")
