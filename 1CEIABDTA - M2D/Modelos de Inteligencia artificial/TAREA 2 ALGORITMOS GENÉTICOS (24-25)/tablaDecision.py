   # Podemos usar la lógica booleana para hacer un código más legible
   # not( a or b ) = not a and not b
def chooseTable(fraccionado:bool, credito:bool, debito:bool):
    if(not (fraccionado or credito or debito)):
        return False
    elif(not (fraccionado or credito) and debito):
        return True
    elif( not fraccionado and credito and debito):
        return False
    elif( fraccionado and not ( credito or debito ) ):
        return False
    elif( fraccionado and not credito and debito):
        return False
    elif( fraccionado and credito and not debito):
        return True
    else:
        return True

def getValue(valueName:str):
    while True:
        valueInput = input(f"Introduce un valor para \"{valueName}\": (0 | 1): ")
        if valueInput.isdigit() and valueInput in ("0","1"):
            return valueInput == "1"
        else:
            print("El valor introducido no es válido")

print( "Aceptado" if chooseTable(getValue("fraccionado"), getValue("credito"), getValue("debito")) else "Denegado" )
    #Si el valor devuelto es True, entonces aceptamos, sino, denegamos