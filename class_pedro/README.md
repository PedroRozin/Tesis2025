# CLASS modificado para obtener las deltas_primas
- Esta carpeta tiene el CLASS (https://github.com/lesgourg/class_public/tree/master/python) modificado para poder obtener las derivadas de las perturbaciones de materia en función de 'a' y para distintos k's (todo en gauge Newtoniano)
- Los cambios importantes están en 'perturbations.c' y 'output.c' y se pueden encontrar buscando la palabra 'acá' en el comando de busqueda ('ctrl.+F' o lo que sea)
- Todas las perturbaciones se guardan en un file llamado 'delta_prime_cdm.txt'. Se genera cuando se corre un .ini y se sobreescribe si se corre varias veces.
- test_delta_prime_gauge_newtonian.ini es el .ini que uso para generar perturbaciones. Modificar a gusto.