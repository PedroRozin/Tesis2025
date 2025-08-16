import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classy import Class
from itertools import product
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#%% Func. generales

def compute_delta_m(delta_cdm, delta_b, omega_cdm, omega_b):
  """
  Compute the total matter density perturbation from CDM and baryonic perturbations.

  Args:
    delta_cdm (float): CDM density perturbation.
    delta_b (float): Baryonic density perturbation.
    omega_cdm (float): Omega_cdm value.
    omega_b (float): Omega_b value.

  Returns:
    array or float128: Total matter density perturbation.
  """
  result = (omega_cdm * delta_cdm + omega_b * delta_b) / (omega_cdm + omega_b)
  if isinstance(result, (int, float)):
    result = np.float128(result)
  else:
    result = np.array(result, dtype='float128')
  return result

def k_horizon(a_ini=.01, omega_m=0.3, omega_r=9.1e-5, c=299792458):
  """Calculate the comoving horizon scale dados los omegas que le pongamos y el a_ini.
  Returns:
    float: The comoving horizon scale in Mpc.
  """
  omega_l= 1-omega_m-omega_r
  k_val = 2 * np.pi * a_ini * 100 / c * np.sqrt(omega_m / a_ini**3 + omega_r / a_ini**4 + omega_l)
  return k_val

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
def percent_diff_vs_class(a_vec, delta_m, a_class, delta_m_class):
    """
    Interpola la curva de CLASS (o cualquiera) sobre los puntos de integración y calcula el porcentaje de diferencia relativa.
    Si hay valores duplicados en a_class, solo toma el primero.
    """
    a_class = np.asarray(a_class)
    delta_m_class = np.asarray(delta_m_class)
    # Solo eliminar duplicados si existen
    if len(np.unique(a_class)) < len(a_class):
        _, unique_indices = np.unique(a_class, return_index=True)
        a_class = a_class[unique_indices]
        delta_m_class = delta_m_class[unique_indices]
    interp_class = interp1d(a_class, delta_m_class, kind='quadratic',
                            bounds_error=False, fill_value="extrapolate")
    delta_m_class_interp = interp_class(a_vec)
    # percent_diff = 100 * (delta_m - delta_m_class_interp) / delta_m_class_interp
    percent_diff = -100 * (delta_m - delta_m_class_interp) / delta_m
    return percent_diff

#%%  Func. de CLASS

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
#%% Func. de file adhoc que sale de perturbations

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

#%% Funciones para obtener resultados desde el output de CLASS 

def set_column_names(df, path):
    """ Set column names for the DataFrame based on the first comment line in the file.
    Renombra a 'tau' if 'tau[Mpc]' is present.
	Args:
		df (DataFrame): The DataFrame to set column names for.
		path (str): Path to the file from which to read the column names.
	Returns:
		DataFrame: The DataFrame with updated column names.
    """
    with open(path) as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                columns = [col.split(':')[1].strip() for col in line[1:].split('  ') if ':' in col]
                df.columns = columns
                break
    if 'tau[Mpc]' in df.columns:
        df.rename(columns={'tau[Mpc]': 'tau'}, inplace=True)
    return df

def read_ini_params(filepath):
    '''
    Obtiene los parameters k, omega_b y omega cdm from an .ini file and returns them as a dictionary.
    Args:
        filepath (str): Path to the .ini file.
    Returns:
        dict: Dictionary containing the parameters.
    '''
    params = {
        "k_output_values": None,
        "Omega_b": None,
        "Omega_cdm": None,
        'h': None,
        'A_s': None,
        'n_s': None,
        'tau_reio': None,
        'YHe': None
    }
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith(";") or line == "": #evito los comentarios y líneas vacías
                continue
            if "=" in line:
                key, value = [x.strip() for x in line.split("=", 1)] #separo valores
                if key in params:
                    params[key] = float(value) #agrego al dicc
    return params

#%% Funciones para integrar numéricamente
Om_r = 9.045385269436243e-05  # Radiation density parameter, hardcoded for now
a0 = 0.01
def Hh(params,a, Om_r =9.045385269436243e-05):
    """
    Calcula el Hubble parameter dado 'a'. Está normalizado a 1 en a=1 (H_0=1). 
    Args:
        params (tuple): A tuple containing the matter density parameter (Om_m_0) and sigma8.
        sigma8 es mudo para las ecuaciones, no se usa.
        Om_r (float): The radiation density parameter. No está definido acá (corregirlo en un futuro con tiempo),
          pero sale del CLASS output con M.Omega_r().
        Om_m_0 (float): The matter density parameter at a=1.
        Om_L (float): The dark energy density parameter at a=1, calculated as
                      Om_L = 1 - Om_m_0 - Om_r.
        a (float or array-like): The scale factor at which to calculate the Hubble parameter.
    Returns:
        float or array-like: The Hubble parameter at the given scale factor 'a'.
    """
    Om_m_0, s8=params
    Om_L=1-Om_m_0-Om_r
    return np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)

def Hh_p(params,a, Om_r = 9.045385269436243e-05):
    """Calcula la derivada del Hubble parameter con respecto a 'a'.
    Args:
        params (tuple): A tuple containing the matter density parameter (Om_m_0) and sigma8.
        sigma8 es mudo para las ecuaciones, no se usa.
        Om_r (float): The radiation density parameter sale del CLASS output con M.Omega_r()
        Om_m_0 (float): The matter density parameter at a=1.
        Om_L (float): The dark energy density parameter at a=1, calculated as
                      Om_L = 1 - Om_m_0 - Om_r.
        a (float or array-like): The scale factor at which to calculate the derivative of the Hubble parameter.
    Returns:
        float or array-like: The derivative of the Hubble parameter with respect to 'a'.
    """
    Om_m_0, s8=params
    Om_L = 1-Om_m_0-Om_r
    num = (3*Om_m_0/a**4+4*Om_r/a**5)
    den = 2*np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)
    return -num/den

def get_delta_cdm_vs_a(params, delta_0,
                        delta_prima_0, a_0=a0, a_f=1 ,method='RK45',
                          atol=1e-10, rtol=1e-8, omr= Om_r):
    """
    Integra la ecuación diferencial para delta_m vs con solve_ivp.
    Permite elegir método y tolerancias.
    Args:
        params: Tuple (Om_m, sigma8)
        delta_0: Valor inicial de delta_m.
        delta_prima_0: Valor inicial de la derivada de delta_m.
        method: Método de solve_ivp ('RK45', 'DOP853', 'Radau', etc.)
        atol: Tolerancia absoluta
        rtol: Tolerancia relativa
    Returns:
        a_vec: Array de scale factors
        delta_num: Array de delta_cdm values
    """
    a_vec = np.linspace(a_0, a_f, 20000)
    def F(a, X):
        f1 = X[1]
        term1 = X[0] * 3 * params[0] / (2 * (Hh(params, a, Om_r=omr) ** 2) * (a ** 5))
        term2 = -X[1] * ((3 / a) + (Hh_p(params, a, Om_r=omr) / Hh(params, a, Om_r=omr)))
        f2 = term1 + term2
        return np.array([f1, f2])
    out2 = solve_ivp(
        fun=F,
        t_span=[a_0, a_f],
        y0=np.array([delta_0, delta_prima_0]),
        t_eval=a_vec,
        method=method,
        atol=atol,
        rtol=rtol
    )
    if not out2.success:
        print(f"Warning: solve_ivp did not converge with method {method}: {out2.message}")
    delta_num = out2.y[0]
    return a_vec, delta_num

#%%