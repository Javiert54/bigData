import os
import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # O prueba 'Qt5Agg' si TkAgg no funciona
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
import traceback # Importar para stack trace
import math # Para la función exponencial

# --- Configuración ---
ENV_NAME: str = 'MountainCarContinuous-v0'
VIDEO_FOLDER: Path = Path('videos') # Carpeta para vídeos

# Parámetros de Entrenamiento
EPISODES: int = 6000
MAX_TIMESTEPS_PER_EPISODE: int = 900
LEARNING_RATE: float = 0.15 # Alpha
DISCOUNT_FACTOR: float = 0.99 # Gamma
INITIAL_EPSILON: float = 1.0
EPSILON_DECAY: float = 0.9994
MIN_EPSILON: float = 0.05
STEP_PENALTY: float = 0.0000 # Penalización base por cada paso
STRONG_ACTION_REWARD_FACTOR: float = 0.1 # Recompensa por acciones fuertes

# --- PARÁMETROS PARA PENALIZACIÓN CONTINUA POR LENTITUD EN EL FONDO ---
BOTTOM_CENTER_POS: float = -0.5  # Posición central del fondo del valle
MAX_BOTTOM_SLOW_PENALTY: float = 1 # Penalización MÁXIMA (en el fondo y parado)
# Sensibilidad: valores más altos -> la penalización disminuye más rápido al alejarse/acelerar
POSITION_PENALTY_SENSITIVITY: float = 60.0 # Sensibilidad a la distancia del fondo
VELOCITY_PENALTY_SENSITIVITY: float = 1100.0 # Sensibilidad a la velocidad (cercanía a cero)


# Parámetros de Discretización
STATE_BINS_COUNT: Tuple[int, int] = (20, 20) # (posición, velocidad)
NUM_ACTIONS: int = 5
ACTION_MAP: List[float] = [-0.7, -0.5, 0.0, 0.5, 0.7]


# --- Parámetros de Grabación de Vídeo ---
RECORD_START_EPISODE: int = 0
RECORD_END_EPISODE: int = EPISODES
RECORD_CHUNK_SIZE: int = 500

# --- Parámetros de Graficación ---
PLOT_START_EPISODE: int = RECORD_START_EPISODE
PLOT_END_EPISODE: int = RECORD_END_EPISODE
MOVING_AVG_WINDOW: int = 50
PLOT_FILENAME_PREFIX: str = "mountaincar_rewards_best_chunk"
# ---------------------

def setup_environment_and_discretization(
    env_name: str,
    state_bins_count: Tuple[int, int],
    render_mode: str = 'rgb_array'
) -> Tuple[gym.Env, List[np.ndarray]]:
    """Crea el entorno y calcula los contenedores para la discretización del estado."""
    env = gym.make(env_name, render_mode=render_mode)
    state_space_low = env.observation_space.low
    state_space_high = env.observation_space.high
    adjusted_high = state_space_high + 1e-6
    state_bins = [
        np.linspace(state_space_low[i], adjusted_high[i], num=state_bins_count[i] + 1)[1:-1]
        for i in range(env.observation_space.shape[0])
    ]
    print("Límites del espacio de estados:", state_space_low, state_space_high)
    print("Contenedores de discretización (bordes internos):")
    for i, bins in enumerate(state_bins):
        print(f"  Dimensión {i}: {len(bins)+1} contenedores")
    return env, state_bins

def initialize_q_table(state_bins_count: Tuple[int, int], num_actions: int) -> np.ndarray:
    """Inicializa la tabla Q con ceros."""
    q_table_shape = state_bins_count + (num_actions,)
    q_table = np.zeros(q_table_shape, dtype=np.float32)
    print(f"Tabla Q inicializada con forma: {q_table.shape}")
    return q_table

def discretize_state(state: np.ndarray, state_bins: List[np.ndarray]) -> Tuple[int, ...]:
    """Discretiza un estado continuo en un índice de tupla discreto."""
    state_idx = tuple(int(np.digitize(state[i], bins)) for i, bins in enumerate(state_bins))
    state_idx = tuple(np.clip(idx, 0, count - 1) for idx, count in zip(state_idx, STATE_BINS_COUNT))
    return state_idx

def choose_action(
    state_idx: Tuple[int, ...],
    q_table: np.ndarray,
    current_epsilon: float,
    num_actions: int
) -> int:
    """Elige una acción usando la política epsilon-greedy."""
    if np.random.random() < current_epsilon:
        action_idx = np.random.randint(0, num_actions)
    else:
        action_idx = np.argmax(q_table[state_idx])
    return action_idx

def run_and_record_episode(
    base_env: gym.Env,
    q_table: np.ndarray,
    state_bins: List[np.ndarray],
    action_map: List[float],
    num_actions: int,
    max_timesteps: int,
    video_folder: Path,
    episode_to_record: int,
    episode_reward: float
) -> None:
    """Ejecuta un episodio con epsilon=0 (explotación) y lo graba."""
    active_env = base_env
    reward_str = f"reward_{episode_reward:.2f}".replace('.', '_')
    print(f"--- Grabando episodio {episode_to_record} (Recompensa: {episode_reward:.2f})... ---")
    try:
        active_env = RecordVideo(
            base_env,
            str(video_folder),
            episode_trigger=lambda e: True,
            name_prefix=f"mountaincar-best-ep{episode_to_record}-{reward_str}"
        )
        active_env.render()
        try:
            initial_observation, info = active_env.reset()
        except Exception as e:
             print(f"Error al resetear el entorno para grabar episodio {episode_to_record}: {e}")
             try: active_env.close()
             except Exception: pass
             return

        state = discretize_state(initial_observation, state_bins)
        terminated = False
        truncated = False
        current_step = 0
        for step in range(max_timesteps):
            action_idx = np.argmax(q_table[state])
            continuous_action = np.array([action_map[action_idx]], dtype=np.float32)
            try:
                next_observation, reward, terminated, truncated, _ = active_env.step(continuous_action)
            except Exception as e:
                print(f"Error durante env.step al grabar episodio {episode_to_record}, paso {step}: {e}")
                terminated = True
            next_state = discretize_state(next_observation, state_bins)
            state = next_state
            current_step = step + 1
            if terminated or truncated: break
        print(f"--- Grabación del episodio {episode_to_record} finalizada (Duración: {current_step} pasos). ---")
    except Exception as e:
        print(f"Error general al inicializar o ejecutar RecordVideo para episodio {episode_to_record}: {e}")
        traceback.print_exc()
    finally:
        if isinstance(active_env, RecordVideo):
             try: active_env.close()
             except Exception as e:
                 print(f"Error al cerrar RecordVideo para episodio {episode_to_record}: {e}")
                 traceback.print_exc()

def train_agent(
    base_env: gym.Env,
    q_table: np.ndarray,
    state_bins: List[np.ndarray],
    action_map: List[float],
    num_actions: int,
    episodes: int,
    max_timesteps: int,
    learning_rate: float,
    discount_factor: float,
    initial_epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    step_penalty: float,
    strong_action_reward_factor: float,
    bottom_center_pos: float,             # <-- Nuevo parámetro
    max_bottom_slow_penalty: float,       # <-- Nuevo parámetro
    position_penalty_sensitivity: float,  # <-- Nuevo parámetro
    velocity_penalty_sensitivity: float,  # <-- Nuevo parámetro
    video_folder: Path,
    record_start_episode: int,
    record_end_episode: int,
    record_chunk_size: int
) -> List[float]:
    """Ejecuta el bucle de entrenamiento Q-learning, grabando el mejor episodio por chunk."""
    rewards_per_episode: List[float] = []
    epsilon = initial_epsilon
    best_reward_in_chunk: float = -np.inf
    best_episode_in_chunk: int = -1
    current_chunk_episode_count: int = 0
    video_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Iniciando Entrenamiento ---")
    print(f"Penalización por paso: {step_penalty}")
    print(f"Recompensa por acción fuerte (factor): {strong_action_reward_factor}")
    # Informar nueva penalización continua
    print(f"Penalización continua por lentitud cerca del fondo (max): {max_bottom_slow_penalty}")
    print(f"  Sensibilidad Posición: {position_penalty_sensitivity}, Velocidad: {velocity_penalty_sensitivity}")
    print(f"Grabación configurada para el mejor episodio cada {record_chunk_size} episodios,")
    print(f"considerando episodios desde {record_start_episode} hasta {record_end_episode}.")

    for episode in range(episodes):
        episode_num_user = episode + 1
        try:
            initial_observation, info = base_env.reset()
        except Exception as e:
             print(f"Error al resetear el entorno base en episodio {episode_num_user}: {e}")
             traceback.print_exc()
             continue

        state = discretize_state(initial_observation, state_bins)
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        for step in range(max_timesteps):
            action_idx = choose_action(state, q_table, epsilon, num_actions)
            continuous_action = np.array([action_map[action_idx]], dtype=np.float32)

            try:
                next_observation, reward, terminated, truncated, _ = base_env.step(continuous_action)
                reward = float(reward) # Recompensa original
                next_state = discretize_state(next_observation, state_bins) # Discretizar ANTES de modificar recompensa

                # --- MODIFICACIÓN DE LA RECOMPENSA ---
                action_value = continuous_action[0]
                position = next_observation[0]
                velocity = next_observation[1]

                # 1. Recompensa por acción fuerte
                reward += strong_action_reward_factor * abs(action_value)

                # 2. Penalización base por paso (si no termina)
                if not (terminated or truncated):
                    reward -= step_penalty

                # 3. Penalización continua por velocidad lenta cerca del fondo
                # Factor basado en posición (1 en el fondo, disminuye al alejarse)
                pos_factor = math.exp(-position_penalty_sensitivity * (position - bottom_center_pos)**2)
                # Factor basado en velocidad (1 a vel 0, disminuye al acelerar)
                vel_factor = math.exp(-velocity_penalty_sensitivity * velocity**2)

                # Penalización = max_penalty * factor_pos * factor_vel
                continuous_penalty = max_bottom_slow_penalty * pos_factor * vel_factor
                reward -= continuous_penalty # Aplicar penalización

                # Descomentar para depuración:
                # if step % 50 == 0 and continuous_penalty > 0.001: # Imprimir si la penalización es significativa
                #    print(f"  Ep {episode_num_user} Step {step}: ContPenalty={continuous_penalty:.4f} (PosF={pos_factor:.3f}, VelF={vel_factor:.3f}) Pos={position:.3f}, Vel={velocity:.4f}")

                # -------------------------------------

            except Exception as e:
                print(f"Error durante env.step en episodio {episode_num_user}, paso {step}: {e}")
                traceback.print_exc()
                terminated = True
                reward = 0.0 # Recompensa neutral en caso de error

            # --- Actualización Q-Table y estado ---
            total_reward += reward # Acumular recompensa modificada

            # Actualización de la Tabla Q
            old_value = q_table[state + (action_idx,)]
            next_max = np.max(q_table[next_state]) if not (terminated or truncated) else 0.0
            target = reward + discount_factor * next_max
            new_value = old_value + learning_rate * (target - old_value)
            q_table[state + (action_idx,)] = new_value

            # Actualizar estado para la siguiente iteración
            state = next_state

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

        # --- Lógica de grabación por chunks ---
        is_within_record_range = record_start_episode <= episode_num_user <= record_end_episode
        if is_within_record_range:
            current_chunk_episode_count += 1
            if total_reward >= best_reward_in_chunk:
                best_reward_in_chunk = total_reward
                best_episode_in_chunk = episode_num_user

            is_chunk_end = (current_chunk_episode_count >= record_chunk_size)
            is_last_episode_in_range = (episode_num_user == record_end_episode)
            is_last_overall_episode = (episode_num_user == episodes)
            should_record_chunk = (is_chunk_end or is_last_episode_in_range or is_last_overall_episode) and best_episode_in_chunk != -1

            if should_record_chunk:
                if record_start_episode <= best_episode_in_chunk <= record_end_episode:
                    run_and_record_episode(
                        base_env=base_env, q_table=q_table, state_bins=state_bins,
                        action_map=action_map, num_actions=num_actions, max_timesteps=max_timesteps,
                        video_folder=video_folder, episode_to_record=best_episode_in_chunk,
                        episode_reward=best_reward_in_chunk
                    )
                best_reward_in_chunk = -np.inf
                best_episode_in_chunk = -1
                current_chunk_episode_count = 0

        # Decaimiento de Epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Imprimir progreso
        if episode_num_user % 100 == 0 or episode_num_user == episodes:
            actual_steps = step + 1
            print(f"Episodio {episode_num_user}/{episodes}: "
                  f"Recompensa: {total_reward:.2f}, "
                  f"Pasos: {actual_steps}, "
                  f"Epsilon: {epsilon:.3f}")

    return rewards_per_episode

def plot_rewards(
    rewards: List[float],
    plot_start_episode: int,
    plot_end_episode: int,
    moving_avg_window: int,
    filename_prefix: str
) -> None:
    """Genera y guarda una gráfica de las recompensas por episodio."""
    total_episodes = len(rewards)
    if total_episodes == 0:
        print("No hay recompensas para graficar.")
        return

    start_index = max(0, plot_start_episode - 1)
    end_index = min(total_episodes, plot_end_episode)

    if start_index >= end_index:
        print(f"\nAdvertencia: Rango de episodios para graficar ({plot_start_episode}-{plot_end_episode}) inválido o vacío.")
        return

    rewards_subset = rewards[start_index:end_index]
    episode_numbers_subset = list(range(start_index + 1, end_index + 1))

    if not episode_numbers_subset:
         print(f"\nAdvertencia: No hay episodios en el rango seleccionado para graficar ({plot_start_episode}-{plot_end_episode}).")
         return

    plt.figure(figsize=(12, 6))
    plt.plot(episode_numbers_subset, rewards_subset, label='Recompensa por Episodio', alpha=0.6, linewidth=0.8)

    if len(rewards) >= moving_avg_window:
        moving_avg_full = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        moving_avg_episodes_full = list(range(moving_avg_window, total_episodes + 1))
        ma_plot_indices = [
            i for i, ep_num in enumerate(moving_avg_episodes_full)
            if plot_start_episode <= ep_num <= plot_end_episode
        ]
        if ma_plot_indices:
            moving_avg_subset = moving_avg_full[ma_plot_indices]
            moving_avg_episodes_subset = [moving_avg_episodes_full[i] for i in ma_plot_indices]
            plt.plot(moving_avg_episodes_subset, moving_avg_subset,
                     label=f'Media Móvil ({moving_avg_window} episodios)', color='red', linewidth=1.5)
        else:
             print(f"Advertencia: La media móvil no tiene puntos dentro del rango de graficación ({plot_start_episode}-{plot_end_episode}).")

    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total')
    plt.title(f'Progreso del Aprendizaje Q-Learning (Episodios {start_index + 1} a {end_index})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_filename = f"{filename_prefix}_ep_{start_index + 1}_to_{end_index}.png"
    try:
        plt.savefig(output_filename)
        print(f"\nGráfica guardada como: '{output_filename}'")
    except Exception as e:
        print(f"\nError al guardar la gráfica '{output_filename}': {e}")
        traceback.print_exc()
    plt.close()

# --- Flujo Principal ---
if __name__ == "__main__":
    print("Iniciando script de Q-Learning para MountainCarContinuous...")
    base_env = None
    try:
        base_env, state_bins = setup_environment_and_discretization(
            ENV_NAME, STATE_BINS_COUNT, render_mode='rgb_array'
        )
    except Exception as e:
        print(f"Error fatal al configurar el entorno: {e}")
        traceback.print_exc()
        exit(1)

    q_table = initialize_q_table(STATE_BINS_COUNT, NUM_ACTIONS)
    all_rewards = []
    try:
        all_rewards = train_agent(
            base_env=base_env,
            q_table=q_table,
            state_bins=state_bins,
            action_map=ACTION_MAP,
            num_actions=NUM_ACTIONS,
            episodes=EPISODES,
            max_timesteps=MAX_TIMESTEPS_PER_EPISODE,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            initial_epsilon=INITIAL_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            min_epsilon=MIN_EPSILON,
            step_penalty=STEP_PENALTY,
            strong_action_reward_factor=STRONG_ACTION_REWARD_FACTOR,
            bottom_center_pos=BOTTOM_CENTER_POS,                   # <-- Pasar nuevo parámetro
            max_bottom_slow_penalty=MAX_BOTTOM_SLOW_PENALTY,       # <-- Pasar nuevo parámetro
            position_penalty_sensitivity=POSITION_PENALTY_SENSITIVITY, # <-- Pasar nuevo parámetro
            velocity_penalty_sensitivity=VELOCITY_PENALTY_SENSITIVITY, # <-- Pasar nuevo parámetro
            video_folder=VIDEO_FOLDER,
            record_start_episode=RECORD_START_EPISODE,
            record_end_episode=RECORD_END_EPISODE,
            record_chunk_size=RECORD_CHUNK_SIZE
        )
        print("\n--- Entrenamiento Completado ---")

        if all_rewards:
            print("\n--- Generando Gráfica de Recompensas ---")
            plot_rewards(
                rewards=all_rewards,
                plot_start_episode=PLOT_START_EPISODE,
                plot_end_episode=PLOT_END_EPISODE,
                moving_avg_window=MOVING_AVG_WINDOW,
                filename_prefix=PLOT_FILENAME_PREFIX
            )
        else:
            print("\n--- No hay recompensas para graficar (posible error durante el entrenamiento) ---")

    except Exception as e:
        print(f"\n--- Error durante el entrenamiento o graficación: {e} ---")
        traceback.print_exc()
    finally:
        if base_env is not None:
            try:
                base_env.close()
                print("\nEntorno base cerrado.")
            except Exception as e:
                print(f"Error al cerrar el entorno base: {e}")
                traceback.print_exc()
        print(f"Vídeos (si se generaron) guardados en: '{VIDEO_FOLDER.resolve()}'")
        print("Script finalizado.")