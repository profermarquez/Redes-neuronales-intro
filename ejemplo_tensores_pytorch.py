import torch
import numpy as np

# 3 filas y 2 columnas
data = [[1,2], 
        [3,4],
        [5,7]]
Tensor = torch.tensor(data)
# Imprimimos nuestro tensor
print(Tensor)

# Imprimimos las forma de nuestro tensor [filas, columnas]
print(Tensor.shape)

# Cambiamos nuestras filas por columnas
new_Tensor = Tensor.reshape([2, 3])
print(Tensor.shape)

# Tambien podemos usar el "-1" de decimos a Pytorch que intuya las dimenciones restantes 
#(estas deben coincidir con la cantidad de datos)
new_Tensor = Tensor.reshape([-1, 3])
print(Tensor.shape)

import torch
import numpy as np

##################### FUNCIONES MATEMATICAS #####################

data_1 = (6,4)
data_2 = (2,4)

# Tensor especifico
Tensor_1 = torch.tensor(data_1)   
Tensor_2 = torch.tensor(data_2)
print(Tensor_1)
print(Tensor_2)

######### Suma #########
Tensor_sum = Tensor_1.add(Tensor_2)
print(Tensor_sum)
# O también
Tensor_sum = Tensor_1 + Tensor_2
print(Tensor_sum)

######### Resta #########
Tensor_sub = Tensor_1.sub(Tensor_2)
print(Tensor_sub)
# O también
Tensor_sub = Tensor_1 - Tensor_2
print(Tensor_sub)

######### Multiplicacion #########
Tensor_mul = Tensor_1.mul(Tensor_2)
print(Tensor_mul)
# O también
Tensor_mul = Tensor_1 * Tensor_2
print(Tensor_mul)

######### Division #########
Tensor_div = torch.div(Tensor_1, Tensor_2)
print(Tensor_div)
# O también
Tensor_div = Tensor_1 / Tensor_2
print(Tensor_div)

##################### FUNCIONES ESPECIALES #####################
######### Tensor Ones #########
# Usamos esta función para crear un tensor que solo contiene unos
# Solo tenemos que definir su forma (filas y columnas)

# Aquí usamos data solo para la forma de nuestro Tensor
forma = (2, 2)
Tensor_uno = torch.ones(forma)
print(Tensor_uno)

######### Tensor Zeros #########
# Usamos esta función para crear un tensor que solo contiene ceros
# Solo tenemos que definir su forma (filas y columnas)

# Aquí usamos data solo para la forma de nuestro Tensor
forma = (2, 2)
Tensor_cero = torch.zeros(forma)
print(Tensor_cero)

######### Tensor Random #########
# Usamos esta función para crear un tensor aleatorio
# Solo tenemos que definir su forma (filas y columnas)

# Aquí usamos data solo para la forma de nuestro Tensor
forma = (2, 2)
Tensor_rand = torch.rand(forma)
print(Tensor_rand)

######### Tensor Narrow #########
# Esta función devuelve una versión reducida de nuestro tensor de entrada

Tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# El primer argumento es el Tensor a modificar (contando el 0)
# El segundo argumento es la dimensión a lo largo de la cual se estrecha (contando el 0)
# El tercer argumento es la dimensión inicial (contando el 0)
# El cuarto y último argumento es la distancia a la dimensión final (contando el 0)

Ejemplo_1 = torch.narrow(Tensor, 0, 0, 2)
print(Ejemplo_1)

Ejemplo_2 = torch.narrow(Tensor, 1, 1, 2)
print(Ejemplo_2)

######### Tensor Tolist #########
# Esta función devuelve el tensor como una lista (anidada)
# Mueve los datos al CPU si es necesario (recordar que las listas no pueden estar en la GPU)
Tensor = torch.randn(2, 2)
print(Tensor[1:2].tolist())

# O también

Lista = Tensor.tolist()
print(Lista[1:2])

######### Tensor Permute #########
# Esta función devuelve nuestro Tensor en el orden de dimensiones que especificamos
Tensor = torch.randn(2, 3, 5)
print(Tensor.size())

# Vamos invertir el orden de dimensiones, la dimensión 3 (2 en realidad por el 0) al principio 
# Y la dimensión 1 (en realidad 0) al final.
Ejemplo_1 = torch.permute(Tensor, (2, 1, 0))
print(Ejemplo_1.size())

Ejemplo_2 = torch.permute(Tensor, (1, 0, 2))
print(Ejemplo_2.size())

######### Tensor Where #########
# Esta función devuelve el Tensor que le damos como entrada SI cumple una condición
# SÍ NO lo rellana  con un valor que especifiquemos
# Estructura: torch.where(condición , Tensor_de_entrada , Relleno)

Tensor = torch.randn(2, 2, dtype=torch.double)
print("Tensor_entrada:\n",Tensor)
print("Tensor Final:\n",torch.where(Tensor > 0, Tensor, 0.))

# También podemos usar de relleno a otro tensor
Tensor_entrada = torch.randn(3, 2)
Relleno = torch.ones(3, 2)
print("Tensor_entrada:\n",Tensor_entrada)
print("Relleno:\n",Relleno)
print("Tensor Final:\n",torch.where(Tensor_entrada > 0, Tensor_entrada, Relleno))

######### Tensor Expand #########
# Devuelve una nueva vista del tensor con dimensiones singleton expandidas a un tamaño mayor
Tensor = torch.tensor([[1], [2], [3]])

# Asi se veria nuestro Tensor
print(Tensor.size())
# Veamos las dimensiones de nuestro Tensor
print(Tensor.size())

# Ahora vamos a expandir sus dimensiones por 3
print(Tensor.expand(3, 4))

# Si le damos un -1 le decimos a pytorch que deje igual la dimensión
# En este caso se ve igual que el anterior porque efectivamente la primera dimensión era 3
print(Tensor.expand(-1, 4))

print(Tensor.expand(3, -1))
