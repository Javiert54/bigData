import os
import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # O Qt5Agg si no va TkAgg
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo
from pathlib import Path
import traceback
import math

# --- Configuración ---
ENV_NAME = 'MountainCarContinuous-v0'
VIDEO_FOLDER = Path('videos') # Carpeta para vídeos

# Parámetros del Training
EPISODES = 6000
MAX_STEPS = 900 # Máximos pasos por juego
LR = 0.15 # Learning Rate (Alpha)
GAMMA = 0.99 # Discount Factor
EPSILON = 1.0 # Empezar explorando mucho
EPSILON_DECAY = 0.9994 
MIN_EPSILON = 0.05 # Mínimo de exploración
STEP_PENALTY = 0.0 # Castigo por cada paso
STRONG_ACTION_REWARD = 0.02 

# --- Castigo por estar quieto cerca del fondo ---
BOTTOM_POS = -0.5  # Dónde está el valle
MAX_PENALTY_BOTTOM = 1 # Castigo MAX si estás abajo y quieto
POS_SENSITIVITY = 60.0 # Sensibilidad a la posición
VEL_SENSITIVITY = 1100.0 # Sensibilidad a la velocidad (si es casi 0)

# Discretizar el estado
NUM_BINS = (10, 10) # Cajas para (posición, velocidad)

ACTIONS = [-0.7, -0.5 -0.3, 0.0, 0.3, 0.5, 0.7] # Qué fuerza aplicamos en cada acción
NUM_ACTIONS = len(ACTIONS) # Cuántas acciones podemos hacer
# --- Para grabar vídeos ---
RECORD_START = 0
RECORD_END = EPISODES
RECORD_EVERY = 500 # Grabar el mejor de cada X episodios

# --- Para la gráfica ---
PLOT_START = RECORD_START
PLOT_END = RECORD_END
AVG_WINDOW = 50 # Para suavizar la gráfica
PLOT_NAME = "grafica_recompensas_montaña"
# ---------------------


def preparar_entorno(env_name, num_bins):
    """Crea el entorno y calcula las divisiones para el estado."""
    env = gym.make(env_name, render_mode='rgb_array') # 'rgb_array' para poder grabar
    low = env.observation_space.low
    high = env.observation_space.high
    
    # Añadimos un poquito al límite alto para que np.linspace funcione bien
    high_adj = high + 1e-6 
    
    # Calculamos los bordes de las cajas para cada dimensión (pos, vel)
    state_bins = []
    for i in range(env.observation_space.shape[0]):
        # linspace crea puntos equidistantes, [1:-1] quita el primero y el último (los bordes exteriores)
        bins = np.linspace(low[i], high_adj[i], num=num_bins[i] + 1)[1:-1]
        state_bins.append(bins)
        
    print("Límites del estado:", low, high)
    print("Divisiones para discretizar (bordes internos):")
    for i, b in enumerate(state_bins):
        print(f"  Dimensión {i}: {len(b)+1} cajas")
    return env, state_bins

# Función para crear la tabla Q (donde guardamos lo que aprende)
def crear_q_table(num_bins, num_actions):
    """Inicializa la tabla Q con ceros."""
    q_table_size = num_bins + (num_actions,) # Tamaño: (cajas_pos, cajas_vel, num_acciones)
    q_table = np.zeros(q_table_size) 
    print(f"Tabla Q creada con tamaño: {q_table.shape}")
    return q_table

# Función para convertir el estado (pos, vel) a índices de la tabla Q
def discretizar_estado(estado, state_bins):
    """Convierte un estado continuo a una tupla de índices (caja_pos, caja_vel)."""
    indices = []
    for i in range(len(estado)):
        # digitize nos dice en qué caja cae el valor
        idx = int(np.digitize(estado[i], state_bins[i]))
        indices.append(idx)
    
    # Asegurarnos de que los índices no se salgan de la tabla (por si acaso)
    indices_clipped = []
    for i in range(len(indices)):
        # clip limita el valor entre 0 y num_bins[i]-1
        clipped_idx = np.clip(indices[i], 0, NUM_BINS[i] - 1) 
        indices_clipped.append(clipped_idx)
        
    return tuple(indices_clipped)

# Función para decidir qué acción tomar
def elegir_accion(estado_idx, q_table, epsilon, num_actions):
    """Elige acción: a veces al azar (explorar), a veces la mejor (explotar)."""
    # Política epsilon-greedy
    if np.random.random() < epsilon:
        # Explorar: elige una acción al azar
        accion_idx = np.random.randint(0, num_actions)
    else:
        # Explotar: elige la mejor acción según la tabla Q para este estado
        accion_idx = np.argmax(q_table[estado_idx])
    return accion_idx

# Función para jugar un episodio y grabarlo si es el mejor
def jugar_y_grabar(env, q_table, state_bins, actions, num_actions, max_steps, video_folder, ep, reward):
    """Juega un episodio usando solo lo aprendido (sin explorar) y lo graba."""
    
    # Preparamos el nombre del vídeo
    reward_str = f"reward_{reward:.2f}".replace('.', '_') 
    filename = f"mountaincar-mejor-ep{ep}-{reward_str}"
    print(f"--- Grabando episodio {ep} (Recompensa: {reward:.2f})... ---")
    
    # Envolvemos el entorno con RecordVideo
    try:
        # Usamos lambda e: True para grabar este episodio específico
        record_env = RecordVideo(env, str(video_folder), episode_trigger=lambda e: True, name_prefix=filename)
        record_env.render() # Necesario para que grabe algo
        
        # Empezamos el episodio
        obs, _ = record_env.reset()
        estado = discretizar_estado(obs, state_bins)
        terminado = False
        truncado = False # Para gym > 0.26
        pasos = 0
        
        # Bucle del episodio (solo explotación)
        for t in range(max_steps):
            accion_idx = np.argmax(q_table[estado]) # Elegir la mejor acción
            accion_continua = np.array([actions[accion_idx]], dtype=np.float32)
            
            # Ejecutar la acción
            try:
                obs_siguiente, rec, terminado, truncado, _ = record_env.step(accion_continua)
            except Exception as e:
                print(f"Error en step grabando ep {ep}, paso {t}: {e}")
                terminado = True # Salir si hay error
                
            estado_siguiente = discretizar_estado(obs_siguiente, state_bins)
            estado = estado_siguiente
            pasos = t + 1
            
            if terminado or truncado:
                break
                
        print(f"--- Grabación ep {ep} terminada ({pasos} pasos). ---")
        
    except Exception as e:
        print(f"Error al preparar o grabar el vídeo del episodio {ep}: {e}")
        traceback.print_exc()
    finally:
        # Importante cerrar el entorno de grabación para que guarde el vídeo
        if 'record_env' in locals() and isinstance(record_env, RecordVideo):
             try:
                 record_env.close()
             except Exception as e:
                 print(f"Error cerrando RecordVideo para ep {ep}: {e}")

# Función principal de entrenamiento
def entrenar(env, q_table, state_bins, actions, num_actions, episodes, max_steps, 
             lr, gamma, start_epsilon, eps_decay, min_epsilon, 
             step_penalty, strong_action_reward,
             bottom_pos, max_penalty_bottom, pos_sensitivity, vel_sensitivity, # Params nuevos
             video_folder, record_start, record_end, record_every):
    """Bucle principal de Q-learning."""
    
    todas_recompensas = [] # Lista para guardar recompensa de cada episodio
    epsilon = start_epsilon
    
    # Para grabar el mejor de cada "chunk" (grupo) de episodios
    mejor_recompensa_chunk = -np.inf
    mejor_episodio_chunk = -1
    episodios_en_chunk = 0
    
    # Crear carpeta de vídeos si no existe
    video_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Empezando Entrenamiento ---")
    print(f"Castigo por paso: {step_penalty}")
    print(f"Premio acción fuerte: {strong_action_reward}")
    print(f"Castigo por lento abajo (max): {max_penalty_bottom}")
    print(f"  Sensibilidad Pos: {pos_sensitivity}, Vel: {vel_sensitivity}")
    print(f"Grabando el mejor episodio cada {record_every} episodios ({record_start}-{record_end}).")

    # Bucle principal de episodios
    for ep in range(episodes):
        ep_num = ep + 1 # Para que empiece en 1
        
        # Reiniciar el entorno para un nuevo episodio
        try:
            obs, _ = env.reset()
        except Exception as e:
             print(f"Error reseteando entorno en episodio {ep_num}: {e}")
             traceback.print_exc()
             continue # Saltar este episodio si falla el reset

        estado = discretizar_estado(obs, state_bins)
        recompensa_total = 0.0
        terminado = False
        truncado = False # Para gym > 0.26
        pasos = 0

        # Bucle dentro de un episodio (pasos)
        for t in range(max_steps):
            # Elegir acción (explorar o explotar)
            accion_idx = elegir_accion(estado, q_table, epsilon, num_actions)
            accion_continua = np.array([actions[accion_idx]], dtype=np.float32)

            # Ejecutar la acción en el entorno
            try:
                obs_siguiente, recompensa, terminado, truncado, _ = env.step(accion_continua)
                recompensa = float(recompensa) # Asegurarse de que es float
                estado_siguiente = discretizar_estado(obs_siguiente, state_bins) # Discretizar ANTES de cambiar recompensa

                # --- MODIFICAR LA RECOMPENSA (Reward Shaping) ---
                accion_valor = accion_continua[0]
                posicion = obs_siguiente[0]
                velocidad = obs_siguiente[1]

                # 1. Premio por usar acciones fuertes (más energía)
                recompensa += strong_action_reward * abs(accion_valor)

                # 2. Castigo base por cada paso (si no ha terminado)
                if not (terminado or truncado):
                    recompensa -= step_penalty

                # 3. Castigo si va lento cerca del fondo del valle
                # Factor por posición (1 si está justo en el fondo, 0 lejos)
                factor_pos = math.exp(-pos_sensitivity * (posicion - bottom_pos)**2)
                # Factor por velocidad (1 si está parado, 0 si va rápido)
                factor_vel = math.exp(-vel_sensitivity * velocidad**2)
                
                # Castigo = MaxCastigo * factor_pos * factor_vel
                castigo_continuo = max_penalty_bottom * factor_pos * factor_vel
                recompensa -= castigo_continuo # Restar el castigo

            except Exception as e:
                print(f"Error en step episodio {ep_num}, paso {t}: {e}")
                traceback.print_exc()
                terminado = True # Salir del episodio si hay error
                recompensa = 0.0 # Recompensa neutra si falla

            # --- Actualizar la Tabla Q ---
            recompensa_total += recompensa # Acumular recompensa (ya modificada)

            # Fórmula Q-learning: Q(s,a) = Q(s,a) + lr * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
            valor_antiguo = q_table[estado + (accion_idx,)]
            
            # El valor futuro es 0 si el episodio termina aquí
            valor_max_siguiente = np.max(q_table[estado_siguiente]) if not (terminado or truncado) else 0.0
            
            # Calculamos el "objetivo" (target) hacia el que queremos mover el valor Q
            objetivo = recompensa + gamma * valor_max_siguiente
            
            # Actualizamos el valor Q para el estado y acción actuales
            valor_nuevo = valor_antiguo + lr * (objetivo - valor_antiguo)
            q_table[estado + (accion_idx,)] = valor_nuevo

            # Pasamos al siguiente estado
            estado = estado_siguiente
            pasos = t + 1 # Guardamos cuántos pasos dimos

            if terminado or truncado:
                break # Salir del bucle de pasos si termina el episodio

        todas_recompensas.append(recompensa_total)

        # --- Lógica para grabar el mejor vídeo del chunk ---
        esta_en_rango_grabacion = record_start <= ep_num <= record_end
        if esta_en_rango_grabacion:
            episodios_en_chunk += 1
            # Si este episodio es mejor que el mejor que teníamos en este chunk
            if recompensa_total >= mejor_recompensa_chunk:
                mejor_recompensa_chunk = recompensa_total
                mejor_episodio_chunk = ep_num

            # Comprobar si hemos terminado un chunk o es el último episodio
            fin_chunk = (episodios_en_chunk >= record_every)
            ultimo_ep_rango = (ep_num == record_end)
            ultimo_ep_total = (ep_num == episodes)
            
            # Si debemos grabar (y hemos encontrado algún mejor episodio en el chunk)
            if (fin_chunk or ultimo_ep_rango or ultimo_ep_total) and mejor_episodio_chunk != -1:
                 # Asegurarnos de que el mejor episodio está dentro del rango permitido
                 if record_start <= mejor_episodio_chunk <= record_end:
                    jugar_y_grabar(
                        env=env, q_table=q_table, state_bins=state_bins,
                        actions=actions, num_actions=num_actions, max_steps=max_steps,
                        video_folder=video_folder, ep=mejor_episodio_chunk,
                        reward=mejor_recompensa_chunk
                    )
                 # Resetear para el siguiente chunk
                 mejor_recompensa_chunk = -np.inf
                 mejor_episodio_chunk = -1
                 episodios_en_chunk = 0

        # Reducir epsilon (menos exploración a medida que aprendemos)
        epsilon = max(min_epsilon, epsilon * eps_decay)

        # Imprimir progreso cada 100 episodios o al final
        if ep_num % 100 == 0 or ep_num == episodes:
            print(f"Episodio {ep_num}/{episodes}: "
                  f"Recompensa: {recompensa_total:.2f}, "
                  f"Pasos: {pasos}, "
                  f"Epsilon: {epsilon:.3f}")

    return todas_recompensas

# Función para dibujar la gráfica de recompensas
def dibujar_grafica(recompensas, start_ep, end_ep, avg_window, filename):
    """Genera y guarda una gráfica de las recompensas."""
    num_episodios = len(recompensas)
    if num_episodios == 0:
        print("No hay recompensas para dibujar.")
        return

    # Ajustar índices para el slicing (Python empieza en 0)
    start_idx = max(0, start_ep - 1)
    end_idx = min(num_episodios, end_ep)

    if start_idx >= end_idx:
        print(f"\nRango de episodios para gráfica ({start_ep}-{end_ep}) no válido.")
        return

    # Seleccionar el trozo de recompensas y los números de episodio correspondientes
    recompensas_sub = recompensas[start_idx:end_idx]
    episodios_sub = list(range(start_idx + 1, end_idx + 1))

    if not episodios_sub:
         print(f"\nNo hay episodios en el rango ({start_ep}-{end_ep}) para la gráfica.")
         return

    plt.figure(figsize=(10, 5)) # Tamaño de la figura
    # Dibujar la recompensa de cada episodio (un poco transparente)
    plt.plot(episodios_sub, recompensas_sub, label='Recompensa Episodio', alpha=0.5, linewidth=1)

    # Calcular y dibujar la media móvil para suavizar la curva
    if len(recompensas) >= avg_window:
        # Usamos convolve para calcular la media móvil de todas las recompensas
        media_movil_total = np.convolve(recompensas, np.ones(avg_window)/avg_window, mode='valid')
        # Los episodios correspondientes a la media móvil (empiezan desde avg_window)
        episodios_media_total = list(range(avg_window, num_episodios + 1))
        
        # Filtrar para mostrar solo la media móvil dentro del rango de la gráfica
        media_movil_sub = []
        episodios_media_sub = []
        for i, ep_num in enumerate(episodios_media_total):
            if start_ep <= ep_num <= end_ep:
                media_movil_sub.append(media_movil_total[i])
                episodios_media_sub.append(ep_num)
                
        if episodios_media_sub: # Si hay puntos de media móvil en el rango
            plt.plot(episodios_media_sub, media_movil_sub, 
                     label=f'Media Móvil ({avg_window} ep)', color='red', linewidth=2)
        # else:
             # print(f"La media móvil no cae en el rango de la gráfica ({start_ep}-{end_ep}).")

    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total')
    plt.title(f'Recompensas Q-Learning (Episodios {start_idx + 1} a {end_idx})')
    plt.legend() # Mostrar leyenda (los labels que pusimos en plot)
    plt.grid(True, linestyle=':') # Poner una rejilla punteada
    plt.tight_layout() # Ajustar para que no se corten los ejes
    
    # Guardar la gráfica
    output_file = f"{filename}_ep_{start_idx + 1}_to_{end_idx}.png"
    try:
        plt.savefig(output_file)
        print(f"\nGráfica guardada como: '{output_file}'")
    except Exception as e:
        print(f"\nError guardando la gráfica '{output_file}': {e}")
        # traceback.print_exc()
    plt.close() # Cerrar la figura para liberar memoria

# --- Código Principal ---
if __name__ == "__main__":
    print("Empezando script Q-Learning para MountainCarContinuo...")
    
    env = None # Inicializar por si falla la creación
    try:
        # 1. Preparar entorno y discretización
        env, state_bins = preparar_entorno(ENV_NAME, NUM_BINS)
        
        # 2. Crear tabla Q
        q_tabla = crear_q_table(NUM_BINS, NUM_ACTIONS)
        
        # 3. Entrenar al agente
        recompensas = entrenar(
            env=env,
            q_table=q_tabla,
            state_bins=state_bins,
            actions=ACTIONS,
            num_actions=NUM_ACTIONS,
            episodes=EPISODES,
            max_steps=MAX_STEPS,
            lr=LR,
            gamma=GAMMA,
            start_epsilon=EPSILON,
            eps_decay=EPSILON_DECAY,
            min_epsilon=MIN_EPSILON,
            step_penalty=STEP_PENALTY,
            strong_action_reward=STRONG_ACTION_REWARD,
            bottom_pos=BOTTOM_POS,                   
            max_penalty_bottom=MAX_PENALTY_BOTTOM,       
            pos_sensitivity=POS_SENSITIVITY, 
            vel_sensitivity=VEL_SENSITIVITY, 
            video_folder=VIDEO_FOLDER,
            record_start=RECORD_START,
            record_end=RECORD_END,
            record_every=RECORD_EVERY
        )
        print("\n--- Entrenamiento Terminado ---")

        # 4. Dibujar gráfica si hay recompensas
        if recompensas:
            print("\n--- Creando Gráfica ---")
            dibujar_grafica(
                recompensas=recompensas,
                start_ep=PLOT_START,
                end_ep=PLOT_END,
                avg_window=AVG_WINDOW,
                filename=PLOT_NAME
            )
        else:
            print("\n--- No se generaron recompensas (quizás hubo un error antes) ---")

    except Exception as e:
        print(f"\n--- ¡ERROR! Algo salió mal: {e} ---")
        traceback.print_exc() # Imprimir el error detallado
    finally:
        # 5. Cerrar el entorno al final (importante!)
        if env is not None:
            try:
                env.close()
                print("\nEntorno cerrado.")
            except Exception as e:
                print(f"Error al cerrar el entorno: {e}")
        
        # Informar dónde se guardaron los vídeos
        print(f"Vídeos (si los hay) en: '{VIDEO_FOLDER.resolve()}'")
        print("Script terminado.")
