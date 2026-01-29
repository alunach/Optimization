# Optimization – Entorno Científico en Docker (Ubuntu 24.04 + C++23 + OpenBLAS + LAPACK/LAPACKE)

Repositorio base:
🔗 [https://github.com/alunach/Optimization.git](https://github.com/alunach/Optimization.git)

Este README describe cómo configurar un entorno **reproducible y académico** usando **Docker + Ubuntu 24.04 (LTS)** para compilar y ejecutar los proyectos de optimización numérica del repositorio, manteniendo la estructura por módulos (`*_project/`, `common/`, `build/`) y habilitando soporte para **C++23**, **OpenBLAS**, **LAPACK** y **LAPACKE**.

Este entorno está diseñado para trabajos prácticos, análisis experimental y comparación de algoritmos como:

* Gradient Descent (GD)
* SGD / Mini-batch SGD
* Nesterov Accelerated Gradient (NAG)
* Adam
* Nesterov Accelerated Look-Ahead (NALA)

---

## 1. Requisitos Previos

### Sistemas Soportados

* Windows 10/11
* Linux
* macOS

### Software necesario

* Docker Desktop
* Git

---

## 2. Instalación de Docker

### Windows / macOS

1. Descargar Docker Desktop:
   [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Instalar y reiniciar si es necesario
3. Verificar instalación:

```bash
docker --version
```

### Ubuntu Linux

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

Verificar:

```bash
docker --version
```

---

## 3. Crear Imagen Docker (Ubuntu 24.04 + Toolchain Científico)

Se recomienda usar Ubuntu 24.04 LTS para estabilidad académica y compatibilidad con bibliotecas numéricas.

### 3.1 Crear archivo `Dockerfile`

En una carpeta vacía del host, crea un archivo llamado `Dockerfile`:

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["/bin/bash"]
```

---

## 4. Construir la Imagen

Desde la carpeta donde se encuentra el `Dockerfile`:

```bash
docker build -t optimization-env .
```

Verificar que la imagen fue creada:

```bash
docker images
```

---

## 5. Ejecutar Contenedor y Montar Carpeta de Trabajo

### Windows (PowerShell)

```powershell
docker run --rm -it --name optimization -v ${PWD} optimization-env
```

### Linux / macOS

```bash
docker run --rm -it --name optimization -v ${PWD} optimization-env
```

Esto crea:

* Contenedor: `optimization-container`
* Carpeta montada: `/workspace`

---

## 6. Clonar Repositorio y Mantener la Estructura del Proyecto

Dentro del contenedor:

```bash
git config --global http.sslVerify false
git clone https://github.com/alunach/Optimization.git/

cd Optimization
```

La estructura esperada debe ser similar a:

```
Optimization/
├── adam_project/
├── autodiff_project/
├── common/
├── gradient_project/
├── knn_project/
├── minibatch_sgd_project/
├── nesterov_project/
├── nesterov_nala_project/
├── pca_project/
├── sgd_project/
├── svd_project/
├── CMakeLists.txt
└── build/   (generado)
```

---

## 7. Verificar Dependencias en el Contenedor

### Compilador

```bash
g++ --version
```

### CMake

```bash
cmake --version
```

### OpenBLAS

```bash
pkg-config --modversion openblas
```

### LAPACK / LAPACKE

```bash
pkg-config --modversion lapacke
```

---

## 8. Ajustes Recomendados en `CMakeLists.txt`

Para habilitar **C++23** en todo el proyecto, edita el archivo raíz `CMakeLists.txt` y cambia:

```cmake
set(CMAKE_CXX_STANDARD 17)
```

por:

```cmake
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### Verificación de NALA

Asegúrate de que el target del proyecto NALA apunte a la carpeta correcta:

```cmake
add_executable(nesterov_nala_project
    nesterov_nala_project/src/main.cpp
    nesterov_nala_project/src/nala.cpp
    nesterov_nala_project/src/quadratic.cpp
)
```

Y que incluya:

```cmake
target_include_directories(nesterov_nala_project PRIVATE
    nesterov_nala_project/include
    common/include
)
```

---

## 9. Compilación del Proyecto Completo

Desde la raíz del repositorio:

```bash
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

Alternativa sin Ninja:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Los ejecutables se generarán dentro de `build/`, por ejemplo:

```bash
./nesterov_nala_project
./adam_project
./sgd_project
```

---

## 10. Ejecución y Generación de Resultados (CSV)

Cada ejecutable genera un archivo CSV con el formato:

```
iter,f,grad_norm,time_ms
```

Se recomienda mover los resultados a:

```
data/
├── convexa/
├── mal_condicionada/
└── no_convexa/
```

Ejemplo:

```bash
mkdir -p data/mal_condicionada
mv nala.csv data/mal_condicionada/
```

---

## 11. Generación de Gráficos para el Paper SBC

Los archivos CSV pueden convertirse en gráficos de convergencia usando Python:

```bash
python3 plot_results.py data/mal_condicionada/*.csv
```

Esto generará:

* `convergencia.png`
* `grad_norm.png`
* `tiempo_vs_iter.png`

Estos archivos pueden ser insertados directamente en el informe LaTeX (SBC).

---

## 12. Control del Contenedor

### Detener

```bash
docker stop optimization-container
```

### Reanudar

```bash
docker start -ai optimization-container
```

### Eliminar

```bash
docker rm optimization-container
docker rmi optimization-env
```

---

## 13. Buenas Prácticas Académicas

* Mantener la lógica interna de los optimizadores sin modificar
* Cambiar únicamente la matriz A o la función objetivo en `main.cpp`
* Documentar cada experimento en `README.md`
* Guardar CSV, capturas y gráficos como evidencia
* Referenciar papers primarios en el informe SBC

---

## 14. Referencias Técnicas

* OpenBLAS — [https://www.openblas.net/](https://www.openblas.net/)
* LAPACK — [https://www.netlib.org/lapack/](https://www.netlib.org/lapack/)
* CMake — [https://cmake.org/](https://cmake.org/)
* Docker — [https://www.docker.com/](https://www.docker.com/)
* Repositorio Optimization — [https://github.com/alunach/Optimization](https://github.com/alunach/Optimization)

---

📌 Autor: Alexander
🎓 Ingeniería de Software
📅 Año: 2026
