#pragma once

#include "io.h"
#include "matrix.h"

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale);  

Image sobel_x(Image src_image);

Image sobel_y(Image src_image);

Image unsharp(Image src_image);

Image gray_world(Image src_image);

Image resize(Image src_image, double scale);

Image custom(Image src_image, Matrix<double> kernel);

Image autocontrast(Image src_image, double fraction);

Image gaussian(Image src_image, double sigma, int radius);

Image gaussian_separable(Image src_image, double sigma, int radius);

Image median(Image src_image, int radius);

Image median_linear(Image src_image, int radius);

Image median_const(Image src_image, int radius);

Image canny(Image src_image, int threshold1, int threshold2);

//my functions
typedef std::tuple<uint, uint, uint> pixel;
typedef std::tuple<double, double, double> dpixel;
typedef Matrix<uint> uMatrix;
typedef Matrix<double> dMatrix;

double get_item(dMatrix&, int, uint, uint);

Image delete_borders(Image &src_image, uint th1, uint th2);

void print_pixel(pixel p);

uint gray(pixel p);

uMatrix to_uMatrix(Image img);

Image to_Image(dMatrix &matr, double koef);

Image to_Image(uMatrix &matr, uint koef);

float MSE(uMatrix &matr1, uMatrix &matr2, int x_offset, int y_offset);

float cross(uMatrix &matr1, uMatrix &matr2, int x_offset, int y_offset);

Image combine(uMatrix &R_channel, uMatrix &G_channel, uMatrix &B_channel, int min_off, int max_off, bool mse);

Image get_mirror(Image &src_image, uint radius_x, uint radius_y);

uMatrix labelling(uMatrix matr);

Image resize(Image &src_image, int new_rows, int new_cols);
