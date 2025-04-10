import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # O prueba 'Qt5Agg' si TkAgg no funciona
import matplotlib.pyplot as plt
import os
# Importa RecordVideo
from gym.wrappers import RecordVideo

# Crea la carpeta para los vídeos si no existe
video_folder = 'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# --- Parámetros ---
episodes = 500
timesteps = 800 # Aumentado para dar más tiempo a llegar a la meta
learning_rate = 0.0005 # Ajustado para Q-learning discreto
discount_factor = 0.99 # Gamma debe estar entre 0 y 1
epsilon = 1.0  # Para exploración (epsilon-greedy)
epsilon_decay = 0.995
min_epsilon = 0.1

# --- Discretización del Espacio de Estados ---
# (El espacio de acciones también necesita discretización para Q-learning estándar)
state_bins_count = [20, 20]  # Número de contenedores por dimensión de estado
env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
state_space_low = env.observation_space.low
state_space_high = env.observation_space.high

# Ajustar límites para que np.digitize funcione correctamente
state_space_high[0] = env.observation_space.high[0] + 1e-6 # Posición
state_space_high[1] = env.observation_space.high[1] + 1e-6 # Velocidad

state_bins = [
    np.linspace(state_space_low[0], state_space_high[0], num=state_bins_count[0] + 1)[1:-1],
    np.linspace(state_space_low[1], state_space_high[1], num=state_bins_count[1] + 1)[1:-1]
]

def discretize_state(state):
    """Discretiza un estado continuo en un índice discreto."""
    state_idx = []
    for i, value in enumerate(state):
        # np.digitize devuelve el índice del bin al que pertenece el valor (1-based)
        # Restamos 1 si usamos los bins directamente, pero con linspace[1:-1] es más simple
        state_idx.append(np.digitize(value, state_bins[i]))
    return tuple(state_idx)

# --- Discretización del Espacio de Acciones ---
# MountainCarContinuous tiene acción continua [-1.0, 1.0]
# Para Q-learning, necesitamos acciones discretas. Vamos a crear 3 acciones:
# 0: Empujar Izquierda (-1.0), 1: No empujar (0.0), 2: Empujar Derecha (1.0)
num_actions = 3
action_map = [-1.0, 0.0, 1.0]

# --- Inicialización de la Tabla Q ---
# Dimensiones: (pos_bins, vel_bins, num_actions)
q_table_shape = tuple(state_bins_count) + (num_actions,)
q_table = np.random.uniform(low=-1, high=0, size=q_table_shape)

# --- Función para elegir acción (Epsilon-Greedy) ---
def choose_action(state_idx, current_epsilon):
    if np.random.random() < current_epsilon:
        # Acción aleatoria (exploración)
        action_idx = np.random.randint(0, num_actions)
    else:
        # Mejor acción según Q-table (explotación)
        action_idx = np.argmax(q_table[state_idx])
    return action_idx

# --- Parámetros de Grabación ---
record_every_n_episodes = 50 # Grabar cada N episodios

# --- Bucle de Entrenamiento ---
rewards = []
# Crea el entorno base una vez fuera del bucle
plot_from_episode = episodes  # Episodio inicial (inclusive, numeración de usuario)
plot_to_episode = 0
for episode in range(episodes):
    current_env = env # Usar el entorno base por defecto
    record_this_episode = (plot_from_episode<=episode<=plot_to_episode) and ((episode % record_every_n_episodes == 0) or (episode == episodes - 1)) # Grabar cada N y el último

    if record_this_episode:
        print(f"Grabando episodio {episode + 1}...")
        # Envuelve el entorno base con RecordVideo para este episodio
        current_env = RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda e: True, # Grabar este episodio específico
            name_prefix=f"mountaincar-episode-{episode+1}"
        )
        # Importante: RecordVideo maneja su propio 'reset' y 'step' internamente
        # Necesitamos obtener el estado inicial del entorno envuelto
        # Nota: env.reset() devuelve (observation, info), tomamos solo observation [0]
        initial_observation, initial_info = current_env.reset()
        state = discretize_state(initial_observation)

    else:
         # Si no grabamos, reseteamos el entorno base normalmente
         initial_observation, initial_info = current_env.reset()
         state = discretize_state(initial_observation)



    total_reward = 0
    terminated = False
    truncated = False
    step_count = 0 # Contador de pasos por episodio

    while not terminated and not truncated:
        # Elige la acción usando el índice discreto
        action_idx = choose_action(state, epsilon)
        # Mapea el índice de acción discreto a la acción continua requerida por el entorno
        continuous_action = np.array([action_map[action_idx]], dtype=np.float32)

        # Ejecuta el paso en el entorno actual (base o envuelto para grabación)
        next_observation, reward, terminated, truncated, _ = current_env.step(continuous_action)
        next_state = discretize_state(next_observation)

        total_reward += reward
        step_count += 1

        # --- Actualización de la Tabla Q ---
        # (Usando índices de estado discretos y índice de acción discreto)
        old_value = q_table[state + (action_idx,)]
        next_max = np.max(q_table[next_state]) # Mejor valor Q para el siguiente estado

        # Fórmula de Q-learning
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state + (action_idx,)] = new_value

        state = next_state

        # Comprobación adicional por si el wrapper de vídeo no trunca correctamente
        if step_count >= timesteps:
             truncated = True # Forzar truncamiento si excede los timesteps máximos

    # --- Fin del Episodio ---
    rewards.append(total_reward)

    # Decaimiento de Epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon) # Asegura que no baje de min_epsilon

    # Cierra el entorno envuelto si se grabó este episodio
    # Esto es crucial para que el vídeo se guarde correctamente
    if record_this_episode:
         current_env.close() # Cierra el wrapper RecordVideo

    print(f"Episodio {episode + 1}: Recompensa Total: {total_reward:.2f}, Pasos: {step_count}, Epsilon: {epsilon:.3f}")


# --- Gráfica de Recompensas (Rango Específico de Episodios) ---
plt.figure(figsize=(12, 6))

# --- Especifica el rango de episodios que quieres graficar ---
# Nota: Los episodios se numeran desde 0 hasta (total_episodes - 1) internamente
#       Pero para el usuario, es más intuitivo pensar desde 1 hasta total_episodes.
#       Ajustaremos los índices para que coincidan con la numeración interna (base 0).
plot_from_episode = 325  # Episodio inicial (inclusive, numeración de usuario)
plot_to_episode = 373    # Episodio final (inclusive, numeración de usuario)
# -------------------------------------------------------------

total_episodes = len(rewards)

# Ajustar a índices base 0 y validar el rango
# El índice inicial es plot_from_episode - 1
start_index = max(0, plot_from_episode - 1)
# El índice final para slicing es plot_to_episode (exclusivo), así que usamos plot_to_episode
# Asegurarse de no exceder el número total de episodios
end_index = min(total_episodes, plot_to_episode)

# Validar que el inicio no sea mayor que el fin después de los ajustes
if start_index >= end_index:
    print(f"\nAdvertencia: El rango de episodios especificado ({plot_from_episode}-{plot_to_episode}) no es válido o no contiene episodios entrenados. No se generará la gráfica.")
else:
    # Selecciona el rango especificado de las recompensas
    rewards_subset = rewards[start_index:end_index]
    # Crea el rango correcto de números de episodio para el eje X (numeración de usuario)
    episode_numbers_subset = range(start_index + 1, end_index + 1) # +1 para mostrar numeración de usuario

    # Grafica solo el subconjunto de recompensas
    plt.plot(episode_numbers_subset, rewards_subset, label='Recompensa por Episodio', alpha=0.6)

    # --- Media móvil (ajustada para el subconjunto) ---
    window_size = 50
    if total_episodes >= window_size:
        # Calcula la media móvil sobre TODAS las recompensas
        moving_avg_full = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        # Los números de episodio correspondientes a la media móvil completa (numeración de usuario)
        moving_avg_episodes_full = range(window_size, total_episodes + 1)

        # Encuentra los índices de la media móvil que caen dentro del rango especificado
        ma_indices_in_range = [
            i for i, ep_num in enumerate(moving_avg_episodes_full)
            if start_index + 1 <= ep_num <= end_index # Comparar con numeración de usuario
        ]

        if ma_indices_in_range:
            # Selecciona el subconjunto de la media móvil y sus episodios correspondientes
            start_ma_index = ma_indices_in_range[0]
            end_ma_index = ma_indices_in_range[-1] + 1 # +1 para slicing exclusivo
            moving_avg_subset = moving_avg_full[start_ma_index:end_ma_index]
            moving_avg_episodes_subset = moving_avg_episodes_full[start_ma_index:end_ma_index]

            plt.plot(moving_avg_episodes_subset, moving_avg_subset, label=f'Media Móvil ({window_size} episodios)', color='orange', linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total')
    # Actualiza el título para reflejar el rango
    plt.title(f'Progreso del Aprendizaje (Episodios {start_index + 1} a {end_index})')
    plt.legend()
    plt.grid(True)

    # Guarda la figura en un archivo PNG
    output_filename = f"mountaincar_rewards_ep_{start_index + 1}_to_{end_index}.png"
    plt.savefig(output_filename)
    print(f"\nGráfica guardada como: '{output_filename}'")

    # Cierra la figura para liberar memoria
    plt.close()




# --- Cierre del Entorno Base ---
env.close()

print("\nEntrenamiento completado.")
print(f"Vídeos guardados en la carpeta: '{video_folder}'")
