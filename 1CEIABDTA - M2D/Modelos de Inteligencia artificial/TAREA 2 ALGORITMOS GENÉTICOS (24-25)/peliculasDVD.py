import random

class Pelicula:
    def __init__(self, titulo: str, genero: str, pesoGB: float):
        self.titulo = titulo
        self.genero = genero
        self.pesoGB = pesoGB

    def __str__(self):
        return f"{self.titulo} - {self.genero} - {self.pesoGB}GB"

peliculas = (
    Pelicula('La última casa a la izquierda', 'TERROR', 1.830),
    Pelicula('Saw IV', 'TERROR', 1.435),
    Pelicula('La huérfana', 'TERROR', 2.163),
    Pelicula('Furia de Titanes', 'ACCIÓN', 1.746),
    Pelicula('El hombre de Acero', 'ACCIÓN', 0.964),
    Pelicula('Los Vengadores', 'ACCIÓN', 2.032),
    Pelicula('American Pie: El reencuentro', 'COMEDIA', 1.746),
    Pelicula('El lado Bueno de las Cosas', 'COMEDIA', 3.139),
    Pelicula('Los tres Chiflados', 'COMEDIA', 0.750),
    Pelicula('Jugada Salvaje', 'SUSPENSE', 2.275),
    Pelicula('El Cuerpo', 'SUSPENSE', 2.082),
    Pelicula('15 años y un día', 'SUSPENSE', 2.321)
)

def fitness(peliculas, combinacion: str, tamaño_dvd: float, restricciones_genero: tuple):
    """
    Calcula qué tan adecuado es un "individuo" (una combinación de películas) basado en el tamaño de DVD y restricciones de género. 
    Penaliza combinaciones que excedan el tamaño del DVD o que se mezclen ciertos géneros o títulos específicos.
    USO: fitness_score = fitness(peliculas, "000001001010", (("COMEDIA", "TERROR")))
    """
    if len(combinacion) != len(peliculas):
        return f"ERROR: La longitud de la combinación no coincide con la cantidad de películas. \n Longitud combinación: {len(combinacion)} Cantidad de películas:  {len(peliculas)}"
    tamaño_total = 0
    generos = set()
    titulos = set()
    for bit in combinacion:
        if bit == "0":
            continue
        tamaño_total += peliculas[bit].pesoGB
        generos.add(peliculas[bit].genero)
        titulos.add(peliculas[bit].titulo)
    
    if tamaño_total > tamaño_dvd:
        return 0  # Penaliza si excede el tamaño del DVD

    if len(titulos & set(["Jugada Salvaje", "El Cuerpo", "Furia de Titanes", "El hombre de Acero"])) > 1:  # Si tenemos más de uno de los títulos especificados
        return 0  # Penaliza si se mezclan títulos específicos
        
    for restriccion in restricciones_genero:
        if restriccion[0] in generos and restriccion[1] in generos:
            return 0  # Penaliza si se mezclan géneros restringidos

    return tamaño_total

def poblacionInicial(tamano_poblacion: int):
    # Genera una población inicial de individuos aleatorios.
    poblacion = tuple(random.sample(range(len(peliculas)), random.randint(1, len(peliculas))) for _ in range(tamano_poblacion))
    return poblacion  # Devuelve la población inicial

def cruce(padre1: str, padre2: str):
    """
    Realiza el cruce entre dos individuos (padre1 y padre2) para generar un nuevo individuo (hijo).
    """
    punto_cruce = random.randint(1, len(padre1) - 1)
    hijo = padre1[:punto_cruce] + padre2[punto_cruce:]
    return hijo

def mutacion(individuo: str, tasa_mutacion: float):
    """
    Realiza la mutación de un individuo con una cierta tasa de mutación.
    """
    individuo_mutado = list(individuo)
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            individuo_mutado[i] = '1' if individuo_mutado[i] == '0' else '0'
    return ''.join(individuo_mutado)



