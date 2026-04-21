import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classy import Class
from itertools import product
from tqdm import tqdm
from funciones_tesis import common_settings
import gc
import traceback # Para ver el error sin romper el loop

def k_horizon(a_ini=.01, omega_m=0.3, omega_r=9.1e-5, c=299792.458):
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
  - h

  Pipeline (TO DO: actualizar esto. quedó MUY desactualizado este pipeline):
  0. Armar un for loop con los parámetros a barrer.
  1. Crear universo dado un conjunto de parámetros con `common_settings`.
  2. get_perturbations() para obtener las perturbaciones de ese universo.
  3. Leer el archivo de texto con `read_adhoc_txt` para obtener las perturbaciones y sus derivadas.
     - Filtrar por a_ini: quedarme con el primer valor de a (el más cercano al a_ini).
  4. Calcula k_horizon() para obtener el k de la escala de horizonte. Esto está en h/Mpc.
  5. Filtra el DataFrame para obtener solo las perturbaciones con k mayor o igual a k_horizon.
  6. Empezar iteración en los k's que quedaron.
  7. Aplica deriv_tau_to_a() para obtener las derivadas respecto a 'a'.
  8. Obtener sigma8 con 'get_sigma8()'.
  9. Armar diccionario con 'Omega_cdm', 'Omega_b', 'A_s', h, 'k_horizon',
    'sigma8', 'delta_cdm', 'delta_b', 'delta_prime_cdm', 'delta_prime_b'. Appendearlos en una lista
  10. Limpiar la memoria de CLASS con `M.struct_cleanup()`.
  11. Borrar el archivo adhoc para poder generear un nuevo en la próxima iteración.
  12. Guardar el diccionario en un DataFrame y exportarlo a un archivo CSV.


  """

  #==========================
  # carpeta donde se guardará todo
  path_folder = '/home/pedrorozin/scripts/outputs_pedro/grillas/'
  n = 'params_para_validacion_v3'

  error_log = f'{path_folder}/{n}/errors_log_{n}.txt' #donde se guardan los errores

  if os.path.exists(path_folder + n):
      raise FileExistsError(f"El directorio {path_folder}/{n} ya existe.")

  if not os.path.exists(path_folder + n):
      os.makedirs(path_folder + n)
  #==========================

  # # rango de valores para cada parámetro
  # omega_m_values = np.arange(0.153, 0.453, 0.033)
  # # A_s_values = np.arange(1.9e-09, 3.e-09, 0.1e-09)
  # h_values = np.arange(0.643, 0.763, 0.033)

  # rango de valores para cada parámetro para validacion
  omega_m_values = np.arange(0.163, 0.443, 0.005)
  # A_s_values = np.arange(1.9e-09, 3.e-09, 0.1e-09)
  h_values = np.arange(0.653, 0.743, 0.003)
  
  # A_s_values = np.arange(1.9e-09, 2.3e-09 + 0.3e-09, 0.3e-09)
  # omega_m_values = np.arange(0.30, 0.32, 0.01)
  # A_s_values = np.arange(1.9e-09, 2.3e-09 , 0.3e-09)
  # h_values = np.arange(0.65, 0.67, 0.01)
  # k_values = np.arange(0.02, 0.22, 0.02)
  results = []
  a_ini= 0.03 #z \approx 33
  
  # Archivo de salida para guardar resultados incrementalmente
  output_file = f'{path_folder}/{n}/grilla_results_{n}.csv'
  file_exists = False  # Para controlar si ya tiene header
  
  for omega_m, h in tqdm(product(omega_m_values, h_values)):
    
    try:
      # 1. Crear universo dado un conjunto de parámetros con `common_settings`.
      M = common_settings(k=0.1, omega_m=omega_m, h=h) #acá parece que es el omega chiquito, pero es Omega grande.

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
      k_hor = k_horizon(a_ini= a_ini_actual, omega_m=omega_m, omega_r=9.1e-5, c=3e5) #c en km/s, k_hor en h/Mpc
      df['k h'] = df['k']/h # k a h/Mpc
      #drop all kh>0.25
      # df = df[df['k h'] <= 0.25]
      
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
              'A_s': 2.e-9,  # A_s fijo
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
      
      # Guardar resultados de esta iteración inmediatamente
      if results:
        df_temp = pd.DataFrame(results)
        df_temp.to_csv(output_file, mode='a', header=not file_exists, index=False)
        file_exists = True
        results = []  # Limpiar lista para liberar memoria
        
    except Exception as e:
      err_msg = f"Error for omega_m={omega_m}, h={h}: {str(e)}\n"
      print(f'\n{err_msg}')
      with open(error_log, 'a') as f:
          f.write(err_msg)
          f.write(traceback.format_exc() + "\n")
          
    finally:  #limpiar todo
      try:
        M.struct_cleanup()
      except:
        pass
      if M in locals():
        del M
  

    # # 9. Limpiar la memoria de CLASS.
    # M.struct_cleanup()
    # del M
    
    # 10. Borrar el archivo adhoc para poder generear un nuevo en la próxima iteración.
    if os.path.exists('/home/pedrorozin/scripts/delta_prime_cdm.txt'):
      os.remove('/home/pedrorozin/scripts/delta_prime_cdm.txt')
    gc.collect()
      
  #11. Leer resultados finales para generar estadísticas
  if os.path.exists(output_file):
    df_results = pd.read_csv(output_file)
  else:
    df_results = pd.DataFrame()
    print("Advertencia: No se generaron resultados")
  
  #create info_grilla.txt
  with open(f'{path_folder}/{n}/info_grilla_{n}.txt', 'w') as f:
      f.write(f'Grilla de parámetros:\n')
      f.write(f'Omega_m: min {omega_m_values.min()}, max {omega_m_values.max()}, len {len(omega_m_values)}\n')
      f.write(f'h: min {h_values.min()}, max {h_values.max()}, len {len(h_values)}\n')
      f.write(f'A_s fijo en 2.e-9\n')
      if not df_results.empty:
        f.write(f'Número total de puntos en la grilla: {len(df_results)}\n')
        f.write(f'min a_ini: {df_results["a"].min()}, max a_ini: {df_results["a"].max()}\n')
        f.write(f'Rango de k_horizon: min {df_results["k_horizon"].min()}h/Mpc, max {df_results["k_horizon"].max()}h/Mpc\n')
      else:
        f.write(f'No se generaron resultados (revisar log de errores)\n')


if __name__ == "__main__":
  main()