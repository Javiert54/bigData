def calcularCombinaciones(numeroDePeliculas):
    #Podemos calcular las combinaciones N bits con el siguiente código:
    result = tuple( format(decimal, 'b').zfill(numeroDePeliculas) for decimal in range(2** (numeroDePeliculas+1) ))
		# con format(decimal, 'b') obtenemos el binario de decimal, y con zfill(numeroDePeliculas) lo rellenamos con ceros a la izquierda
    return result

#print(calcularCombinaciones(12)) # Imprime 8192 combinaciones
class Pelicula:
    def __init__(self, titulo: str, genero: str, pesoGB: float):
        self.titulo = titulo
        self.genero = genero
        self.pesoGB = pesoGB

    def __str__(self):
        return f"{self.titulo} - {self.genero} - {self.peso}GB"

peliculas = [
	Pelicula('La última casa a la izquierda', 'TERROR', 1.830),
    Pelicula('Saw IV', 'TERROR', 1.435),
    Pelicula('La huérfana', 'TERROR', 2.163),
    Pelicula('Furia de Titanes', 'ACCIÓN', 1.746),
    Pelicula('El hombre de Acero', 'ACCIÓN', 0.964),
    Pelicula('Los Vengadores', 'ACCIÓN', 2.032),
    Pelicula('American Pie: El reencuentro', 'COMEDIA', 1.746),
    Pelicula('El lado Bueno de las Cosas', 'COMEDIA', 3.139),
    Pelicula('Los tres Chiflados', 'Comedia', 0.750),
    Pelicula('Jugada Salvaje', 'Suspense', 2.275),
    Pelicula('El Cuerpo', 'SUSPENSE', 2.082),
    Pelicula('15 años y un día', 'SUSPENSE', 2.321)
]