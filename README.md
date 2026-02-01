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

### WSL2

Para instalar WSL2 en Windows 11, abre PowerShell o CMD como administrador y ejecuta wsl --install. 
Esto habilita los componentes necesarios y descarga Ubuntu por defecto. Tras reiniciar el equipo, se configurará un usuario y contraseña de Linux. 

```bash
wsl --install
```

Reinicia tu ordenador cuando se te solicite.

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
3. Importante: En la ventana de configuración, asegúrate de que la casilla "Use the WSL 2 based engine" (Usar el motor basado en WSL 2) esté marcada
4. Verificar instalación:

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
docker run --rm -it --name optimization-container -v ${PWD} optimization-env
```

### Linux / macOS

```bash
docker run --rm -it --name optimization-container -v ${PWD} optimization-env
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
├── conjugate_gradient_project/
├── knn_project/
├── data/
├── data_output/
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

## 8. Compilación del Proyecto Completo

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

## 9. Ejecución y Generación de Resultados (CSV)

Si estas en la carpeta raiz el repositorio dentro del contenedor ejecuta:
```bash
cd build
```

Ejecutar los los comandos:

```bash
./gradient_project 0
./gradient_project 1
./gradient_project 2
./gradient_project 3
./nesterov_project 0
./nesterov_project 1
./nesterov_project 2
./nesterov_project 3
./adam_project 0
./adam_project 1
./adam_project 2
./adam_project 3
./amsgrad_project 0
./amsgrad_project 1
./amsgrad_project 2
./amsgrad_project 3
./nesterov_nala_project 0
./nesterov_nala_project 1
./nesterov_nala_project 2
./nesterov_nala_project 3
./conjugate_gradient_project 0
./conjugate_gradient_project 1
./conjugate_gradient_project 2
./conjugate_gradient_project 3
```

El parámetros corresponen a casos propuestos que modifican la matriz A:
1. 0 = Convexo
2. 1 = Fuertemente Convexo
3. 2 = Mal acondicionada
4. 3 = No Convexo

Cada ejecutable genera archivos CSV con el formato:

```
iter,f,grad_norm,time_ms
```

Con la siguioente estructura en el repositorio:

```
data_ouput/
├── adam_convex.csv/
├── adam_illcond.csv/
├── adam_nonconvex.csv/
├── adam_strong.csv/
├── .../
```

---

## 10. Generación de Gráficos para el Paper SBC

Los archivos CSV pueden convertirse en gráficos de convergencia usando Python:

1. Buscar la carpeta python en el repositorio y extraer el copiar el código del archivo main.py.
2. Crear un documento nuevo en Google Colab.
3. Subir los archivos CSV a la ruta /content/optimization/
4. Por cada ejecución necesitarás modificar una línea del código similar a/con a,b,c y d:
```bash
a) paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*_strong.csv"), recursive=True))
b) paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*_convex.csv"), recursive=True))
c) paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*_illcond.csv"), recursive=True))
d) paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*_nonconvex.csv"), recursive=True))
```
5. Se generaran los archivos en la ruta /content

Las 4 ejecuciones generarán:

* `comparativo_optimization_illcond.png`
* `comparativo_optimization_convex.png`
* `comparativo_optimization_strong.png`
* `comparativo_optimization_nonconvex.png`

Estos archivos pueden ser insertados directamente en el informe LaTeX (SBC).

---

## 11. Control del Contenedor

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

## 12. Buenas Prácticas Académicas

* Mantener la lógica interna de los optimizadores sin modificar
* Cambiar únicamente la matriz A o la función objetivo en `main.cpp`
* Documentar cada experimento en `README.md`
* Guardar CSV, capturas y gráficos como evidencia
* Referenciar papers primarios en el informe SBC

---

## 13. Referencias Técnicas

* OpenBLAS — [https://www.openblas.net/](https://www.openblas.net/)
* LAPACK — [https://www.netlib.org/lapack/](https://www.netlib.org/lapack/)
* CMake — [https://cmake.org/](https://cmake.org/)
* Docker — [https://www.docker.com/](https://www.docker.com/)
* Repositorio Optimization — [https://github.com/alunach/Optimization](https://github.com/alunach/Optimization)

---

📌 Autor: Alexander
🎓 Ingeniería de Software
📅 Año: 2026
