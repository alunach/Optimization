cmake_minimum_required(VERSION 3.10)
project(optimized_ml_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Directorios comunes
include_directories(common/include)
include_directories(knn_project/include)  # Agregar directorio para kNN
include_directories(pca_project/include)  # Agregar directorio para PCA
include_directories(autodiff_project/include)  # Agregar directorio para Autodiff
include_directories(svd_project/include) # REduce img
include_directories(gradient_project/include)
include_directories(minibatch_sgd_project/include)
include_directories(sgd_project/include)
include_directories(adam_project/include)
include_directories(nesterov_project/include)
include_directories(nesterov_nala_project/include)


# Subproyecto: kNN
add_executable(knn_project
    knn_project/src/main.cpp
    knn_project/src/knn.cpp
    common/src/read_csv.cpp
    common/src/write_csv.cpp
    common/src/plot_utils.cpp
)

# Enlace de bibliotecas para kNN
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
#find_package(LAPACKE REQUIRED)
set(LAPACKE_LIB "/usr/lib/x86_64-linux-gnu/liblapacke.so")

target_link_libraries(knn_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Subproyecto: PCA
add_executable(pca_project
    pca_project/src/main.cpp
    pca_project/src/pca_lapack.cpp
    common/src/read_csv.cpp
    common/src/write_csv.cpp
    common/src/plot_utils.cpp
)

# Enlace de bibliotecas para PCA
target_link_libraries(pca_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Subproyecto: AutoDiff
add_executable(autodiff_project
    autodiff_project/src/main.cpp
    autodiff_project/src/autodiff.cpp
)

# Enlace de bibliotecas para AutoDiff
target_link_libraries(autodiff_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


add_executable(svd_project 
    svd_project/src/main.cpp 
    svd_project/src/SVDImage.cpp
)

target_link_libraries(svd_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIB})


add_executable(gradient_project 
    gradient_project/src/main.cpp 
    gradient_project/src/GradientDescent.cpp
)

target_link_libraries(gradient_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


add_executable(minibatch_sgd_project 
    minibatch_sgd_project/src/main.cpp 
    minibatch_sgd_project/src/MiniBatchSGD.cpp
)

target_link_libraries(minibatch_sgd_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


add_executable(sgd_project 
    sgd_project/src/main.cpp 
    sgd_project/src/SGD.cpp
)

target_link_libraries(sgd_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


add_executable(adam_project 
    adam_project/src/main.cpp 
    adam_project/src/Adam.cpp
)

target_link_libraries(adam_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


add_executable(nesterov_project 
    nesterov_project/src/main.cpp 
    nesterov_project/src/Nesterov.cpp
)

target_link_libraries(nesterov_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(nesterov_nala_project 
    nesterov_project/src/main.cpp 
    nesterov_project/src/nala.cpp 
    nesterov_project/src/quadratic.cpp
)

target_link_libraries(nesterov_nala_project ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
