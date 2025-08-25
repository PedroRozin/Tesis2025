import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classy import Class
from itertools import product
from tqdm import tqdm

def common_settings(k=0.01, omega_m=.3, A_s=2.e-9, h=0.68): 
  """
  Set common settings for the CLASS simulation.
  Esto hace basicamente lo mismo que el diccionario de common_settings del principio (el de Julien),
    pero con los valores de k, omega_cdm, A_s y h como argumentos.
  Args:
    k (float): Value of k for the simulation.
    omega_cdm (float): Omega_cdm value for the simulation.
    A_s (float): A_s value for the simulation.
    h (float): h value for the simulation.

  Returns:
    A dictionary containing the common settings for the simulation.
  """
  _common_settings = {
    'output': 'mPk',
    'k_output_values': k,
    'h': h,
    # 'Omega_b': 0.3-omega_cdm,
    # 'Omega_cdm': omega_cdm,
    'Omega_m': omega_m,
    'A_s': A_s,
    'n_s': 0.965,
    'tau_reio': 0.05430842,
    'YHe': 0.2454,
    'compute damping scale': 'yes',
    'gauge': 'newtonian'
    }
  M = Class()
  M.set(_common_settings)
  M.compute()
  return M

def k_horizon(a_ini=.01, omega_m=0.3, omega_r=9.1e-5, c=299792458):
  """Calculate the comoving horizon scale dados los omegas que le pongamos y el a_ini.
  Returns:
    float: The comoving horizon scale in Mpc.
  """
  omega_l= 1-omega_m-omega_r
  k_val = 2 * np.pi * a_ini * 100 / c * np.sqrt(omega_m / a_ini**3 + omega_r / a_ini**4 + omega_l)
  return k_val

def read_adhoc_txt(file_path = '/home/pedrorozin/scripts/delta_prime_cdm.txt'):
  """
  Función que parece simple, pero es clave. Trae todo lo que queremos a menos de la derivada respecto a 'a'.
  
  Args:
    file_path (str): Path to the text file.
  
  Returns:
    DataFrame: df con deltas y deltas dot (cdm y barionica) y a, k y H. NO incluye las derivadas respecto a 'a'; somente a tau.
  """
  # Especificar dtype para mayor precisión en las perturbaciones
  dtype_dict = {
    'delta_cdm': 'float128',
    'delta_dot_cdm': 'float128', 
    'delta_b': 'float128',
    'delta_dot_b': 'float128',
    'a': 'float64',
    'k': 'float64',
    'H': 'float64'
  }
  
  df = pd.read_csv(file_path, sep=' ', names=['delta_cdm', 'delta_dot_cdm', 'delta_b', 'delta_dot_b', 'a', 'k', 'H'], dtype=dtype_dict)
  return df

def deriv_tau_to_a(df, column_name='delta_dot_cdm'):
    """
    apply chain rule to convert tau to a.
    d delta/da = d delta/dtau * d tau/da = dot(delta)/(H a)
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if column_name == 'delta_dot_cdm':
      df['delta_prime_cdm'] = (df[column_name] / (df['H'] * df['a'])).astype('float128')
      return df
    if column_name == 'delta_dot_b':
      df['delta_prime_b'] = (df[column_name] / (df['H'] * df['a'])).astype('float128')
      return df
    
def get_sigma8(M):
  """
  Extracts the current value of sigma8 from the perturbations dictionary.

  Args:
    dicc: The perturbations dictionary (default: all_k['scalar'][0]).

  Returns:
    The current value of sigma8.
  """

  _sigma= M.get_current_derived_parameters(['sigma8'])
  return _sigma['sigma8']

def compute_delta_m(delta_cdm, delta_b, omega_cdm, omega_b):
  """
  Compute the total matter density perturbation from CDM and baryonic perturbations.

  Args:
    delta_cdm (float): CDM density perturbation.
    delta_b (float): Baryonic density perturbation.
    omega_cdm (float): Omega_cdm value.
    omega_b (float): Omega_b value.

  Returns:
    float: Total matter density perturbation.
  """
  result = (omega_cdm * delta_cdm + omega_b * delta_b) / (omega_cdm + omega_b)
  return np.float128(result)

def main():
  """
  Función principal para armar la grilla. Esto va a devolver los vectores completos; NO SOLO LAS CONDICIONES INICIALES.
  Por ahora, la idea es obtener todo para chequear que esté todo bien después.
  CLASS calcula para todos los k's, entonces ese no es un parámetro a barrer; simplemente filtramos el df para k in horizont.
  Parámetros a barrer:
  - Omega_cdm
  - A_s
  - h

  Pipeline:
  0. Armar un for loop con los parámetros a barrer.
  1. Crear universo dado un conjunto de parámetros con `common_settings`.
  2. get_perturbations() para obtener las perturbaciones de ese universo.
  3. Leer el archivo de texto con `read_adhoc_txt` para obtener las perturbaciones y sus derivadas.
     - Filtrar por a_ini: quedarme con el primer valor de a (el más cercano al a_ini).
  4. Calcula k_horizon() para obtener el k de la escala de horizonte.
  5. Filtra el DataFrame para obtener solo las perturbaciones con k mayor o igual a k_horizon.
  6. Aplica deriv_tau_to_a() para obtener las derivadas respecto a 'a'.
  7. Obtener sigma8 con 'get_sigma8()'.
  8. Armar diccionario con 'Omega_cdm', 'Omega_b', 'A_s', h, 'k_horizon',
    'sigma8', 'delta_cdm', 'delta_b', 'delta_prime_cdm', 'delta_prime_b'. Appendearlos en una lista
  9. Limpiar la memoria de CLASS con `M.struct_cleanup()`.
  10. Borrar el archivo adhoc para poder generear un nuevo en la próxima iteración.
  11. Guardar el diccionario en un DataFrame y exportarlo a un archivo CSV.
  

FILTRAR POR A INI Y CORREGIR ARGUMENTS DEL K THRESHOLD
  """
  # rango de valores para cada parámetro
  omega_m_values = np.arange(0.30, 0.41, 0.01)
  # A_s_values = np.arange(1.9e-09, 2.3e-09 + 0.3e-09, 0.3e-09)
  A_s_values = np.arange(1.9e-09, 3.e-09, 0.1e-09)
  h_values = np.arange(0.65, 0.76, 0.01)
  # omega_m_values = np.arange(0.30, 0.32, 0.01)
  # A_s_values = np.arange(1.9e-09, 2.3e-09 , 0.3e-09)
  # h_values = np.arange(0.65, 0.67, 0.01)
  # k_values = np.arange(0.02, 0.22, 0.02)
  results = []
  a_ini= 0.05
  for omega_m, A_s, h in tqdm(product(omega_m_values, A_s_values, h_values)):
    # 1. Crear universo dado un conjunto de parámetros con `common_settings`.
    M = common_settings(omega_m=omega_m, A_s=A_s, h=h) #acá parece que es el omega chiquito, pero es Omega grande.

    # 2. get_perturbations() para obtener las perturbaciones de ese universo.
    # esto ejecuta el CLASS.compute() y devuelve las perturbaciones en el archivo adhoc.
    _perturbations = M.get_perturbations() #variable muda. solo sirve para ejecutar el compute() de CLASS.
    
    # 3. leer el archivo de texto con `read_adhoc_txt` para obtener las perturbaciones y sus derivadas.
    df = read_adhoc_txt()
    #filtrar df con el valor más cercano de a_ini
    #polemico porque después derivo respecto a 'a' multiplicando 'vectores'; pero es lo mismo (y más rápido) porque lo hace elemento a elemento.
    df = df[df['a'] >= a_ini]
    #sort by a y eliminar duplicados
    df = df.drop_duplicates(subset=['a'], keep='first').sort_values('a')

    # 4. Calcula k_horizon() para obtener el k de la escala de horizonte.
    a_ini_actual = df['a'].min()  # El a mínimo después de filtros iniciales
    k_hor = k_horizon(a_ini= a_ini_actual, omega_m=omega_m, omega_r=9.1e-5, c=3e5) #c en km/s
    df['k h'] = df['k']*h
    
    # 5. Filtra el DataFrame para obtener solo las perturbaciones con k mayor o igual a k_horizon.
    # df_filtered = df[df['k'] >= k_hor].copy()
    df_filtered = df[df['k h'] >= k_hor].copy()

    uniques_ks = df_filtered['k'].unique()
    
    sigma8 = get_sigma8(M)
    
    # 8. Armar diccionario con los resultados.
    omega_b = M.Omega_b()
    _omega_m = M.Omega_m() #debería ser el mismo de la iteración
    omega_cdm = omega_m - omega_b

    for _k in uniques_ks:
        # Filtrar por k específico
        df_k = df_filtered[df_filtered['k'] == _k].copy()
        
        # Aplicar derivadas solo a este k
        df_k = deriv_tau_to_a(df_k, column_name='delta_dot_cdm')
        df_k = deriv_tau_to_a(df_k, column_name='delta_dot_b')
        
        # Obtener el índice del a mínimo para este k
        min_a_idx = df_k['a'].idxmin()
        
        # Extraer valores para este k específico
        delta_cdm = np.float128(df_k.loc[min_a_idx, 'delta_cdm'])
        delta_b = np.float128(df_k.loc[min_a_idx, 'delta_b'])
        delta_m = compute_delta_m(delta_cdm, delta_b, omega_cdm, omega_b)
        delta_prime_cdm = np.float128(df_k.loc[min_a_idx, 'delta_prime_cdm'])
        delta_prime_b = np.float128(df_k.loc[min_a_idx, 'delta_prime_b'])
        delta_prime_m = compute_delta_m(delta_prime_cdm, delta_prime_b, omega_cdm, omega_b)

        result_dict = {
            'a': df_k.loc[min_a_idx, 'a'],  
            'k': df_k.loc[min_a_idx, 'k'],  # k original
            'k h': df_k.loc[min_a_idx, 'k h'],
            'Omega_cdm': omega_cdm,
            'Omega_b': omega_b,
            'Omega_m': omega_m,
            'A_s': A_s,
            'h': h,
            'k_horizon': k_hor,
            'sigma8': sigma8,
            'delta_cdm': delta_cdm,  
            'delta_b': delta_b,      
            'delta_m': delta_m,
            'delta_prime_cdm': delta_prime_cdm,  
            'delta_prime_b': delta_prime_b,
            'delta_prime_m': delta_prime_m
        }
        results.append(result_dict)
        

    # 9. Limpiar la memoria de CLASS.
    M.struct_cleanup()
    
    # 10. Borrar el archivo adhoc para poder generear un nuevo en la próxima iteración.
    os.remove('/home/pedrorozin/scripts/delta_prime_cdm.txt')
  #11. Guardar el diccionario en un DataFrame y exportarlo a un archivo CSV.
  df_results = pd.DataFrame(results)
  df_results.to_csv('grilla_results_x11_2.csv', index=False)

if __name__ == "__main__":
  main()