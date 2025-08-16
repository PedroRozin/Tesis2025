from setuptools import setup, find_packages

setup(
    name="funciones_tesis",
    version="0.1.0",
    description="Funciones para Tesis de Licenciatura 2025 - cosmo",
    author="Pedro Rozin",
    author_email="pedrorozin@hotmail.com",
    packages=find_packages(),
    py_modules=["funciones_tesis"],
    install_requires=[
        "numpy",
        "pandas", 
        "matplotlib",
        "scipy",
        "tqdm",
        "classy"  # Si tienes CLASS instalado
    ],
    python_requires=">=3.8",
)
