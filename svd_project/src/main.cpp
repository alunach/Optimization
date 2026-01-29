#include "SVDImage.h"
#include <iostream>

int main() {
    std::string input_file = "../img/input.pgm";
    std::string output_file = "../img/output.pgm";
 
    int num_singular_values;
    std::cout << "Ingrese el número de valores singulares a conservar: ";
    std::cin >> num_singular_values;

    SVDImage img(input_file);
    img.applySVD(num_singular_values);
    img.saveImage(output_file);

    std::cout << "Imagen reducida guardada en " << output_file << std::endl;
    return 0;
}
