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

def fitness(peliculas, cromosoma: str, tamaño_dvd: float, restricciones_genero: tuple):
    """
    Calcula qué tan adecuado es un "individuo" (una combinación de películas) basado en el tamaño de DVD y restricciones de género. 
    Penaliza combinaciones que excedan el tamaño del DVD o que se mezclen ciertos géneros o títulos específicos.
    USO: fitness_score = fitness(peliculas, "000001001010", (("COMEDIA", "TERROR")))
    """
    if len(cromosoma) != len(peliculas):
        return f"ERROR: La longitud de la combinación no coincide con la cantidad de películas. \n Longitud combinación: {len(cromosoma)} Cantidad de películas:  {len(peliculas)}"
    
    tamaño_total = 0
    generos = set()
    titulos = set()
    for index, bit in enumerate(cromosoma):
        if bit == "0":
            continue
        tamaño_total += peliculas[index].pesoGB
        generos.add(peliculas[index].genero)
        titulos.add(peliculas[index].titulo)
    
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
    poblacion =  tuple("".join(tuple (str(random.randint(0,1)) for _ in range(len(peliculas)))) for _ in range(tamano_poblacion))
    
    return poblacion  # Devuelve la población inicial

def recombinacion(padre1: str, padre2: str):
    """
    Realiza el cruce entre dos individuos (padre1 y padre2) para generar un nuevo individuo (hijo).
    """
    punto_cruce = random.randint(1, len(padre1) - 1)
    hijo = padre1[:punto_cruce] + padre2[punto_cruce:]
    return hijo

#Definimos la función de mutación
def mutacion(individuo: str, tasa_mutacion: float):
    """
    Realiza la mutación de un individuo con una cierta tasa de mutación.
    """
    individuo_mutado = list(individuo)
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            individuo_mutado[i] = '1' if individuo_mutado[i] == '0' else '0'
    return ''.join(individuo_mutado)

#Definimos el algoritmo genético
def algoritmogenetico(peliculas, tamaño_dvd, restricciones_genero, tamano_poblacion, Ngeneraciones, tasa_mutacion):
    """
    Ejecuta el proceso de evolución durante varias generaciones, optimizando los individuos para ajustarse mejor al tamaño de DVD.
    """
    # Generar la población inicial
    poblacion = poblacionInicial(tamano_poblacion)
    while max([fitness(peliculas, individuo, tamaño_dvd, restricciones_genero) for individuo in poblacion]) ==0: #Si la población inicial no cumple el fitness, generamos otra población
        poblacion = poblacionInicial(tamano_poblacion)
    
    for generacion in range(Ngeneraciones):
        # Evaluar la aptitud de cada individuo
        aptitudes = [fitness(peliculas, individuo, tamaño_dvd, restricciones_genero) for individuo in poblacion]
        
        # Seleccionar los mejores individuos para la reproducción
        seleccionados = [poblacion[i] for i in range(len(poblacion)) if aptitudes[i] > 0]
        
        # Generar la nueva población mediante cruce y mutación
        nueva_poblacion = []
        while len(nueva_poblacion) < tamano_poblacion:
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = recombinacion(padre1, padre2)
            hijo = mutacion(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
    
    # Evaluar la aptitud de la población final
    aptitudes_finales = [fitness(peliculas, individuo, tamaño_dvd, restricciones_genero) for individuo in poblacion]
    
    # Encontrar el mejor individuo
    mejor_individuo = poblacion[aptitudes_finales.index(max(aptitudes_finales))]
    mejor_aptitud = max(aptitudes_finales)
    peliculasMejorIndividuo = tuple(peliculas[index] for index, i in enumerate(mejor_individuo) if i == "1")
    return mejor_individuo, mejor_aptitud, peliculasMejorIndividuo


tamaño_dvd = 4.7
restricciones_genero = (("COMEDIA", "TERROR"))
tamano_poblacion = 200
generaciones = 60
tasa_mutacion = 0.09

mejor_individuo, mejor_aptitud, peliculasMejorIndividuo = algoritmogenetico(peliculas, tamaño_dvd, restricciones_genero, tamano_poblacion, generaciones, tasa_mutacion)
print(f"Mejor individuo: {mejor_individuo}")
print(f"Peso total: {mejor_aptitud}GB")
print("Películas del mejor individuo:")
pesoTotal = 0
for pelicula in peliculasMejorIndividuo:
    print(pelicula.__str__())
    pesoTotal += pelicula.pesoGB



