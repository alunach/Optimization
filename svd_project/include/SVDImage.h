#ifndef SVD_IMAGE_H
#define SVD_IMAGE_H

#include <vector>
#include <string>

class SVDImage {
private:
    std::vector<std::vector<double>> image_matrix;
    int width, height;
    
public:
    SVDImage(const std::string& filename);
    void applySVD(int num_singular_values);
    void saveImage(const std::string& filename);
};

#endif
