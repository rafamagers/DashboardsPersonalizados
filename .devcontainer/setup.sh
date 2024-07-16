#!/bin/bash

# Descargar y compilar R en el directorio del usuario
mkdir -p ~/R
cd ~/R
wget https://cran.r-project.org/src/base/R-4/R-4.0.2.tar.gz
tar -xzvf R-4.0.2.tar.gz
cd R-4.0.2

# Configurar e instalar R en el directorio del usuario
./configure --prefix=$HOME/R
make
make install

# Añadir R a la variable PATH
echo 'export PATH=$HOME/R/bin:$PATH' >> ~/.bashrc
echo 'export R_HOME=$HOME/R' >> ~/.bashrc

# Fuente el archivo bashrc para actualizar el PATH
source ~/.bashrc

# Confirmar que R está instalado
R --version