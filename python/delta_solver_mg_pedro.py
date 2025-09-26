import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.constants import c as c_ms
import torch
import joblib
import pandas as pd

from funciones_tesis import ImprovedRegressionNN

# Constantes cosmológicas por defecto
DEFAULT_OM_R_0 = 9.4e-5  # Omega_radiación hoy
# DEFAULT_OM_R_0 = 5.38e-5  # Omega_radiación hoy
DEFAULT_OM_M_0 = 0.305
DEFAULT_C = c_ms/1000  # Velocidad de la luz
DEFAULT_Z_F = 0  # Redshift final
DEFAULT_Z_INI_HS = 20  # Redshift inicial para el modelo HS
DEFAULT_Z_0 = 30  # Redshift inicial para la integración delta (cambiar a 25)
default_b=0.1
default_k=0.01
default_h = .68


class DeltaSolver:
    def __init__(self, Om_m_0=DEFAULT_OM_M_0, Om_r_0=DEFAULT_OM_R_0, z_f=DEFAULT_Z_F,
                z_ini_HS=DEFAULT_Z_INI_HS, z_0=DEFAULT_Z_0, c=DEFAULT_C,b=default_b,k=default_k, h=default_h):
        """
        Inicializa el solucionador con los parámetros cosmológicos
        """
        self.Om_m_0 = Om_m_0
        self.Om_r_0 = Om_r_0
        self.Om_l_0 = 1 - Om_m_0 - Om_r_0
        self.z_f = z_f
        self.z_ini_HS = z_ini_HS
        self.z_0 = z_0
        self.c = c
        self.b = b
        self.k = k
        self.h = h
        self.a_0 = 1/(1 + z_0)  

    def H_LCDM(self,z):#H_cdm/H0
        return np.sqrt(self.Om_r_0 * (1+z)**4 + self.Om_m_0 * (1+z)**3 + self.Om_l_0)


    def dH_dz_lcdm(self,z):#dH_cdm/dz/H0
        H_z_over_H0=self.H_LCDM(z)
        dH=1/2*(3*self.Om_m_0*(1+z)**2 + 4*self.Om_r_0*(1+z)**3)/H_z_over_H0
        return dH
    
# Friedmann equations for f(R)

    @staticmethod
    def Gamma(r, b):
        return (r + b) * ((r + b)**2 - 2*b) / (4*b*r)
    

    
    def F(self, z, X):
        x, y, v, Om, r, Or = X
        denom = (z + 1)
        g = self.Gamma(r, self.b)
        f1 = (-Om - 2*v + x + 4*y + x*v + x**2) / denom
        f2 = -(v*x*g - x*y + 4*y - 2*y*v) / denom
        f3 = -v * (x*g + 4 - 2*v) / denom
        f4 = Om * (-1 + 2*v + x) / denom
        f5 = -r * g * x / denom
        f6 = Or*(2*v+x)/denom
        return np.array([f1, f2, f3, f4, f5, f6])
    
    def condiciones_iniciales(self):
        """Condiciones iniciales en z0"""
        eta = self.Om_m_0 * (1 + self.z_ini_HS)**3 + self.Om_r_0 * (1 + self.z_ini_HS)**4 + self.Om_l_0
        y0 = np.zeros(6)
        y0[0] = 0.0
        y0[1] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 2 * self.Om_l_0) / (2 * eta)
        y0[2] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 4 * self.Om_l_0) / (2 * eta)
        y0[3] = self.Om_m_0 * (1 + self.z_ini_HS)**3 / eta
        y0[4] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 4 * self.Om_l_0) / self.Om_l_0
        y0[5] = self.Om_r_0 * (1 + self.z_ini_HS)**4 / eta
        return y0
    
    def integ_HS(self):
        """Integra las ecuaciones de Friedmann de HS"""
        M = 10000
        z_array = np.linspace(self.z_ini_HS, self.z_f, M)
        y0 = self.condiciones_iniciales()
        
        # Necesitamos pasar self.F como una función que solo toma z, X y b
        def f_wrapper(z, X):
            return self.F(z, X)
        
        #args = (b,)
        sol = solve_ivp(f_wrapper, [self.z_ini_HS, self.z_f], y0, t_eval=z_array, 
                        method='RK45', rtol=1e-12, atol=1e-10)
        
        z_vec = sol.t
        x_vec = sol.y[0]
        v_vec = sol.y[2]
        r_vec = sol.y[4]

        return z_vec, x_vec, v_vec, r_vec
    
    def dH_dz(self, z, r, v, x):
        denom = (z + 1)
        g = self.Gamma(r, self.b)
        dv_dz = -v * (x*g + 4 - 2*v) / denom
        dr_dz = -r * g * x / denom
        coeff1 = 1/4 * (1 - self.Om_m_0)
        coeff2 = np.sqrt(r/(2*v) * (1 - self.Om_m_0))
        dH = coeff1/coeff2 * (dr_dz/v - r/v**2 * dv_dz)
        return dH
    
    def H_HS(self,z_ini_HS=None):
        """Calcula H_HS, dH_HS/dz, R_HS"""
        if z_ini_HS is None:
            z_ini_HS = self.z_ini_HS
            
        z_vec, x_vec, v_vec, r_vec = self.integ_HS()
        H_HS_2 = np.sqrt((r_vec/(2*v_vec)) * (1 - self.Om_m_0))
        dHdz = self.dH_dz(z_vec, r_vec, v_vec, x_vec)
        H_prime = -dHdz * (1+z_vec)**2
        
        # Usar extrapolate en lugar de bounds_error=True
        H_interp = interp1d(z_vec, H_HS_2)
        Hprime_interp = interp1d(z_vec, H_prime)
        r_interp = interp1d(z_vec, r_vec)
        
        return H_interp, Hprime_interp, r_interp

    def Geff(self, z, r_val):
        """Calcula G_efectivo"""
        c = self.c
        h= self.h
        is_scalar = np.isscalar(z)
        z_array = np.atleast_1d(z)
        lamb = 3 * (h*100)**2 * (1 - self.Om_m_0) / c**2  # aca va H0**2 (ahora esta puesto =100 porque k se entra en esas unidades) CAMBIADO
        result = np.zeros_like(z_array)

        for i, z_val in enumerate(z_array):
            r = r_val(z_val) if callable(r_val) else r_val
            FR_num = 1 - 2/(self.b * (1 + r/self.b)**2)
            FRR_num = 4/(lamb * self.b**2 * (1 + r/self.b)**3)
            m = FRR_num/FR_num
            eps = m * self.k**2 * (1+z_val)**2
            result[i] = 1/FR_num * (1 + 1/(3 + 1/eps))

        return result[0] if is_scalar else result
    
    def delta_mg(self, a, y, H_num, H_prime_num, r_num):
        """Ecuación para evolución de delta"""
        z = 1/a - 1
        
        if z > self.z_ini_HS:
            H = self.H_LCDM(z)
            dH = -self.dH_dz_lcdm(z) * (1+z)**2
            term1 = dH/H + 3/a
            term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5)
        else:
            H = H_num(z)
            dH = H_prime_num(z)
            r_val=r_num(z)
            term1 = dH/H + 3/a
            term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5) * self.Geff(z, r_val)
        
        res1 = y[1]
        res2 = -term1 * y[1] + term2 * y[0]
        
        return [res1, res2]
    
    # terminos de la ecuacion
    
    def terminos(self, a, H_num, H_prime_num, r_num):

        term1 = []
        term2 = []
        term3 = []
        term4=  []
        for ai in a:
            zi = 1/ai - 1
            nini=abs(np.log(1e-3))
            if zi > self.z_ini_HS:
                H = self.H_LCDM(zi)
                dH = -self.dH_dz_lcdm(zi) * (1+zi)**2
                term1.append(dH/H + 3/ai)
                term2.append((3 * self.Om_m_0) / (2 * H**2 * ai**5)) 
                term3.append(nini*(dH/H*ai))
                term4.append((nini**2*3*self.Om_m_0)/(2*H**2*ai**3))
            else:
                H = H_num(zi)
                dH = H_prime_num(zi)
                r_val=r_num(zi)
                term1.append(dH/H + 3/ai)
                term2.append((3 * self.Om_m_0) / (2 * H**2 * ai**5) * self.Geff(zi, r_val))
                term3.append(nini*(dH/H*ai))
                term4.append((nini**2*3*self.Om_m_0)/(2*H**2*ai**3)*self.Geff(zi, r_val))
        return np.array(term1), np.array(term2),np.array(term3), np.array(term4)

    def red_para_condiciones_iniciales(self, parametros = [0.03, 0.1, 0.68, 0.3], name ='tanh'):
        """ condiciones iniciales a partir de la red neuronal entrenada"""
        model = ImprovedRegressionNN(activation='tanh')
        folder_path = str(name)
        network_name = f'_{folder_path}'
        path_model = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/regression_model{network_name}.pth'
        path_scaler_X = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_X{network_name}.pkl'
        path_scaler_y = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_y{network_name}.pkl'
        model.load_state_dict(torch.load(path_model))  
        model.eval()
        scaler_X = joblib.load(path_scaler_X)
        scaler_y = joblib.load(path_scaler_y)
        if type(parametros) is dict:
            a = np.array([parametros['a']])
            kh = np.array([parametros['k h']])
            h = np.array([parametros['h']])
            Omega_m = np.array([parametros['Omega_m']])
        elif type(parametros) is list or type(parametros) is np.ndarray:
            a = np.array([parametros[0]])
            kh = np.array([parametros[1]])
            h = np.array([parametros[2]])
            Omega_m = np.array([parametros[3]])
        elif type(parametros) is pd.DataFrame:
            a = np.array([parametros['a'].values[0]])
            kh = np.array([parametros['k h'].values[0]])
            h = np.array([parametros['h'].values[0]])
            Omega_m = np.array([parametros['Omega_m'].values[0]])
        X = np.column_stack((a, kh, h, Omega_m))
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = model(X_tensor).numpy()
        y = scaler_y.inverse_transform(y_scaled)
        delta_ini, delta_p_ini = y[0]
        return [delta_ini, delta_p_ini]
    

    def delta_lcdm(self, a, y):
        z = 1/a - 1
        """Ecuación para evolución de delta"""
        
        H = self.H_LCDM(z)
        dH = -self.dH_dz_lcdm(z) * (1+z)**2
        term1 = dH/H + 3/a
        term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5)

        res1 = y[1]
        res2 = -term1 * y[1] + term2 * y[0]
        
        return [res1, res2]
    
    def solve_delta_lcdm(self, num_points=1000):
        """Resuelve la ecuación delta para parámetros dados"""
        # Preparar vector de factor de escala
        a_ini = 1/self.z_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        a_vec = a_vec[(a_vec > a_ini) & (a_vec < a_fin)]
        a_vec = np.insert(a_vec, 0, a_ini)
        a_vec = np.insert(a_vec, len(a_vec), a_fin)

        # Condiciones iniciales
        # y0 = [a_ini, 1.0]  # Modo creciente en régimen de materia: δ ∝ a
        y0 = self.red_para_condiciones_iniciales(parametros = [a_ini, self.k, self.h, self.Om_m_0], name ='tanh')
        
        # Obtener interpoladores H
        H_num, H_prime_num, r_num = self.H_HS()
        
        # Función wrapper para delta_mg
        def delta_wrapper(a, y):
            return self.delta_lcdm(a, y)
        
        # Resolver la ecuación
        sol_lcdm = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                            t_eval=a_vec, method='RK45',
                            atol=1e-12, rtol=1e-10)
        
        a_num_mg = sol_lcdm.t
        delta_num_mg, delta_p_num_mg = sol_lcdm.y
        
        return a_num_mg, delta_num_mg, delta_p_num_mg
    
    def solve_delta_mg(self, num_points=1000):
        """Resuelve la ecuación delta para parámetros dados"""
        # Preparar vector de factor de escala
        a_ini = 1/self.z_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        a_vec = a_vec[(a_vec > a_ini) & (a_vec < a_fin)]
        a_vec = np.insert(a_vec, 0, a_ini)
        a_vec = np.insert(a_vec, len(a_vec), a_fin)

        # Condiciones iniciales  (pedro modificar)
        # y0 = [a_ini, 1.0]  # Modo creciente en régimen de materia: δ ∝ a
        y0 = self.red_para_condiciones_iniciales(parametros = [a_ini, self.k, self.h, self.Om_m_0], name ='tanh')
        # Obtener interpoladores H
        H_num, H_prime_num, r_num = self.H_HS()
        
        # Función wrapper para delta_mg
        def delta_wrapper(a, y):
            return self.delta_mg(a, y, H_num, H_prime_num, r_num)
        
        # Resolver la ecuación
        sol_MG = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                            t_eval=a_vec, method='RK45',
                            atol=1e-12, rtol=1e-10)
        
        a_num_mg = sol_MG.t
        delta_num_mg, delta_p_num_mg = sol_MG.y
        
        return a_num_mg, delta_num_mg, delta_p_num_mg
    
    def plot_delta(self, a_num_mg, delta_num_mg, delta_p_num_mg):
        """Grafica los resultados de delta"""
        plt.figure(figsize=(10, 6))
        plt.plot(a_num_mg, delta_num_mg, label='Delta MG')
        plt.plot(a_num_mg, delta_p_num_mg, label='Delta prime MG')
        plt.xscale('log')
        plt.xlabel('Factor de escala (a)')
        plt.ylabel('Delta')
        plt.title(f'Evolución de delta con Om_m_0={self.Om_m_0}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    

def calculate_delta_mg(z_ini_HS=25,Om_m_0=0.305, k=0.1, b=0.1, num_points=1000):
    solver = DeltaSolver(z_ini_HS=z_ini_HS,Om_m_0=Om_m_0,b=b,k=k)
    return solver.solve_delta_mg(num_points)

def calculate_delta_lcdm(Om_m_0=0.305, num_points=1000):
    solver = DeltaSolver(Om_m_0=Om_m_0)
    return solver.solve_delta_lcdm(num_points)