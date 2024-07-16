#!/bin/bash

# Directorio de instalación de R
R_DIR=$HOME/R

# Crear el directorio si no existe
mkdir -p $R_DIR
cd $R_DIR

# Descargar y compilar R
wget https://cran.r-project.org/src/base/R-4/R-4.0.2.tar.gz
tar -xzvf R-4.0.2.tar.gz
cd R-4.0.2

# Configurar e instalar R en el directorio del usuario
./configure --prefix=$R_DIR
make
make install

# Añadir R a la variable PATH y definir R_HOME
echo 'export PATH=$HOME/R/bin:$PATH' >> ~/.bashrc
echo 'export R_HOME=$HOME/R' >> ~/.bashrc

# Fuente el archivo bashrc para actualizar el PATH
source ~/.bashrc

# Confirmar que R está instalado
R --version

# Instalar las dependencias de Python
pip3 install --user streamlit rpy2 RyStats pandas

# Confirmar que las dependencias están instaladas
pip3 list | grep -E "streamlit|rpy2|RyStats|pandas"