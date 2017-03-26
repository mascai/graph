/*
    Solved problems:
    1) Removal of film frames 
    2) Gray world
    3) sharpening
    4) Autocontrast
    5) Mirror
*/
#include "align.h"
#include <string>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include <algorithm>

using std::string;
using std::cout;
using std::endl;
using std::tie;
using std::pow;
using std::max;
using std::abs;
using std::min;
using std::sqrt;
using std::exp;
using std::atan2;
using std::set;
using std::vector;
using std::sort;
using std::log;


#define PI 3.14159265359

void print_pixel(pixel p) // print pixel(r, g, b)
{
	uint r, g, b;
	tie(r, g, b) = p;
	cout << r << ' ' << g << ' ' << b << endl;
}

uint gray(pixel p) // pixel -> uint
{
	int r, g, b;
	tie(r, g, b) = p;
    return 0.2125 * float(r) + 0.7154 * float(g) + 0.0721 * float(b);
}

uMatrix get_uMatrix(Image img) // Image -> uMatrix
{
    uMatrix matr(img.n_rows, img.n_cols);
    for (uint r = 0; r < img.n_rows; r++)
        for (uint c = 0; c < img.n_cols; c++)
         matr(r, c) = gray(img(r, c));
    return matr;
}

Image to_Image(dMatrix &matr, double koef=1) // dMatrix -> Image, double -> pixel * koef
{
	uint rows = matr.n_rows, cols = matr.n_cols;
	Image img(rows, cols);
	for (uint r = 0; r < rows; r++)
		for (uint c = 0; c < cols; c++)
			img(r, c) = pixel(matr(r, c) * koef, matr(r, c) * koef, matr(r, c) * koef);
	return img;
}

Image to_Image(uMatrix &matr, uint koef=1) // dMatrix -> Image, double -> pixel * koef
{
	uint rows = matr.n_rows, cols = matr.n_cols;
	Image img(rows, cols);
	for (uint r = 0; r < rows; r++)
		for (uint c = 0; c < cols; c++)
			img(r, c) = pixel(matr(r, c) * koef, matr(r, c) * koef, matr(r, c) * koef);
	return img;
}

// MSE is better!
float MSE(uMatrix &matr1, uMatrix &matr2, int x_offset, int y_offset) // calculate MSE
{
	uint rows = min(matr1.n_rows, matr2.n_rows), cols = min(matr1.n_cols, matr2.n_rows);
	double sum = 0;
	for (int x = max(0, x_offset); x < int(rows) + min(0, x_offset); x++)
		for (int y = max(0, y_offset); y < int(cols) + min(0, y_offset); y++)
			sum += pow(int(matr1(x, y)) - int(matr2(x - x_offset, y - y_offset)), 2);
	return sum / (double(rows) * double(cols));
}

float cross(uMatrix &matr1, uMatrix &matr2, int x_offset, int y_offset) // calculate cross-corellation
{
	uint rows = min(matr1.n_rows, matr2.n_rows), cols = min(matr1.n_cols, matr2.n_rows);
	float sum = 0;
	for (uint x = max(0, x_offset); x < rows + min(0, x_offset); x++)
		for (uint y = max(0, y_offset); y < cols + min(0, y_offset); y++)
			sum += int(matr1(x, y)) * int(matr2(x - x_offset, y - y_offset));
	return sum;
}

// combine image channels
Image combine(uMatrix &R_channel, uMatrix &G_channel, uMatrix &B_channel, int min_off=-15, int max_off=15, bool mse=true) // combine channels to one image
{
    int x_offset_gb=min_off, y_offset_gb=min_off, x_offset_gr=min_off, y_offset_gr=min_off;
	float best_rate_gb = MSE(G_channel, B_channel, min_off, min_off),
          best_rate_gr = MSE(G_channel, R_channel, min_off, min_off);
	for (int x = min_off; x <= max_off; x++)
		for (int y = min_off; y <= max_off; y++)
		{
			float rate_gr = mse ? MSE(G_channel, R_channel, x, y):
                                  cross(G_channel, R_channel, x, y),
                  rate_gb = mse ? MSE(G_channel, B_channel, x, y):
                                  cross(G_channel, B_channel, x, y);
            if ((rate_gb <= best_rate_gb && mse) || (rate_gb > best_rate_gb && not mse))
            {
                best_rate_gb = rate_gb;
                x_offset_gb = x;
                y_offset_gb = y;
            }
            if ((rate_gr <= best_rate_gr && mse) || (rate_gr > best_rate_gr && not mse))
            {
                best_rate_gr = rate_gr;
                x_offset_gr = x;
                y_offset_gr = y;
            }
		}

	uint rows = min(R_channel.n_rows, min(B_channel.n_rows, G_channel.n_rows)),
		 cols = min(R_channel.n_cols, min(B_channel.n_cols, G_channel.n_cols));

    uint start_x = max(0, max(x_offset_gr, x_offset_gb)),
         start_y = max(0, max(y_offset_gr, y_offset_gb)),
         end_x = rows + min(0, min(x_offset_gr, x_offset_gb)),
         end_y = cols + min(0, min(y_offset_gr, y_offset_gb));

    Image new_img(end_x - start_x, end_y - start_y);
    for (uint x = start_x; x < end_x; x++)
        for (uint y = start_y; y < end_y; y++)
            new_img(x - start_x, y - start_y) = pixel(R_channel(x - x_offset_gr, y - y_offset_gr),
                                                      G_channel(x, y),
                                                      B_channel(x - x_offset_gb, y - y_offset_gb));

    return new_img;
}

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{
	uint cut_rows = srcImage.n_rows / 3, cut_cols = srcImage.n_cols;

    // Get channels aka Image
	Image B_Image = srcImage.submatrix(0, 0, cut_rows, cut_cols),
		  G_Image = srcImage.submatrix(cut_rows, 0, cut_rows, cut_cols),
		  R_Image = srcImage.submatrix(cut_rows * 2, 0, cut_rows, cut_cols);

    // Delete borders on each image
	R_Image = delete_borders(R_Image, 10, 45);
	G_Image = delete_borders(G_Image, 10, 45);
	B_Image = delete_borders(B_Image, 10, 45);

	// Get channels aka Matrix<uint> of Image (works faster!)
	uMatrix B_channel = get_uMatrix(B_Image),
			G_channel = get_uMatrix(G_Image),
			R_channel = get_uMatrix(R_Image);

    Image dst_image = combine(R_channel, G_channel, B_channel, -12, 12, 1); //MSE is better!

    if (isPostprocessing)
    {
        if (postprocessingType == "--unsharp")
            dst_image = unsharp(dst_image);
        if (postprocessingType == "--gray-world")
            dst_image = gray_world(dst_image);
        if (postprocessingType == "--autocontrast")
            dst_image = autocontrast(dst_image, fraction);
    }

    return dst_image;
}

Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel);
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}

Image unsharp(Image src_image) {
    dMatrix kernel = {{-1. / 6., -2. / 3., -1. / 6.},
                      {-2. / 3., 13. / 3., -2. / 3.},
                      {-1. / 6., -2. / 3., -1. / 6.}};
    return custom(src_image, kernel);
}

Image gray_world(Image src_image) {
	uint rows = src_image.n_rows, cols = src_image.n_cols;
    float R = 0, G = 0, B = 0;
    for (uint i = 0; i < rows; i ++)
        for (uint j = 0; j < cols; j++)
        {
            float r, g, b;
            tie(r, g, b) = src_image(i, j);
            R += r;
            G += g;
            B += b;
        }
    float S = (R + G + B) / 3.;
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++)
        {
            float r, g, b;
            tie(r, g, b) = src_image(i, j);
            uint new_r = min(255.f, r * S / R), new_g = min(255.f, g * S / G), new_b = min(255.f, b * S / B);
            src_image(i, j) = pixel(new_r, new_g, new_b);
        }
    return src_image;
}

Image resize(Image &src_image, int new_rows, int new_cols)
{
    uint src_rows = src_image.n_rows, src_cols = src_image.n_cols;
    double scale_r = double(new_rows) / double(src_rows),
           scale_c = double(new_cols) / double(src_cols);
    uint dst_rows = double(src_rows) * scale_r, dst_cols = double(src_cols) * scale_c;
    Image dst_image(dst_rows, dst_cols);
    Image mirror = get_mirror(src_image, 2, 2);
    for (uint i = 0; i < dst_rows; i++)
        for (uint j = 0; j < dst_cols; j++)
        {
            uint x = double(i) / scale_r, y = double(j) / scale_c;
            int r, g, b;
            tie(r, g, b) = mirror(x + 2, y + 2);
            for (int xx = -1; xx <= 1; xx+=2)
                for (int yy = -1; yy <= 1; yy+=2)
                {
                    int rr, gg, bb;
                    tie(rr, gg, bb) = mirror(x + 2 + xx, y + 2 + yy);
                    r += rr / 4;
                    g += gg / 4;
                    b += bb / 4;
                }
            dst_image(i, j) = pixel(r / 2, g / 2, b / 2);
        }
    return dst_image;
}

//linear interpolation
Image resize(Image src_image, double scale) {
    uint src_rows = src_image.n_rows, src_cols = src_image.n_cols;
    uint dst_rows = double(src_rows) * scale, dst_cols = double(src_cols) * scale;
    Image dst_image(dst_rows, dst_cols);
    Image mirror = get_mirror(src_image, 2, 2);
    for (uint i = 0; i < dst_rows; i++)
        for (uint j = 0; j < dst_cols; j++)
        {
            uint x = double(i) / scale, y = double(j) / scale;
            int r, g, b;
            tie(r, g, b) = mirror(x + 2, y + 2);
            for (int xx = -1; xx <= 1; xx+=2)
                for (int yy = -1; yy <= 1; yy+=2)
                {
                    int rr, gg, bb;
                    tie(rr, gg, bb) = mirror(x + 2 + xx, y + 2 + yy);
                    r += rr / 4;
                    g += gg / 4;
                    b += bb / 4;
                }
            dst_image(i, j) = pixel(r / 2, g / 2, b / 2);
        }
    return dst_image;
}

dpixel sum(dpixel p1, dpixel p2)
{
	double r1, r2, g1, g2, b1, b2;
	tie(r1, g1, b1) = p1;
	tie(r2, g2, b2) = p2;
	return dpixel(r1 + r2, g1 + g2, b1 + b2);
}

dpixel mul(dpixel p, double k)
{
	double r, g, b;
	tie(r, g, b) = p;
	return dpixel(r * k, g * k, b * k);
}

// returns mirrored image
Image get_mirror(Image &src_image, uint radius_x, uint radius_y)
{
	Image mirror_image(src_image.n_rows + 2 * radius_x, src_image.n_cols + 2 * radius_y);	
	uint rows = src_image.n_rows, cols = src_image.n_cols;
	// center
	for (uint r = radius_x; r < rows + radius_x; r++)
		for (uint c = radius_y; c < cols + radius_y; c++)
			mirror_image(r, c) = src_image(r - radius_x, c - radius_y);
	
	//up
	for (uint r = 0; r < radius_x; r++)
		for (uint c = 0; c < mirror_image.n_cols; c++)
			{
				uint x = radius_x - r;
				uint y;
				if (c < radius_y)
					y = radius_y - c;
				else if (c >= radius_y && c < cols + radius_y)
					y = c - radius_y;
				else
					y = 2 * (cols + radius_y) - c - radius_y - 1;
				mirror_image(r, c) = src_image(x, y);
			}
	//down
	for (uint r = rows + radius_x; r < mirror_image.n_rows; r++)
		for (uint c = 0; c < mirror_image.n_cols; c++)
			{
				uint x = 2 * (rows + radius_x) - r - radius_x - 1;
				uint y;
				if (c < radius_y)
					y = radius_y - c;
				else if (c >= radius_y && c < cols + radius_y)
					y = c - radius_y;
				else
					y = 2 * (cols + radius_y) - c - radius_y - 1;
				mirror_image(r, c) = src_image(x, y);
			}
	//left
	for (uint r = 0; r < mirror_image.n_rows; r++)
		for (uint c = 0; c < radius_y; c++)
			{
				uint y = radius_y - c;
				uint x;
				if (r < radius_x)
					x = radius_x - r;
				else if (r >= radius_x && r < rows + radius_x)
					x = r - radius_x;
				else
					x = 2 * (rows + radius_x) - r - radius_x - 1;
				mirror_image(r, c) = src_image(x, y);
			}
	//right
	for (uint r = 0; r < mirror_image.n_rows; r++)
		for (uint c = cols + radius_y; c < mirror_image.n_cols; c++)
			{
				uint y = 2 * (cols + radius_y) - c - radius_y - 1;
				uint x;
				if (r < radius_x)
					x = radius_x - r;
				else if (r >= radius_x && r < rows + radius_x)
					x = r - radius_x;
				else
					x = 2 * (rows + radius_x) - r - radius_x - 1;
				mirror_image(r, c) = src_image(x, y);
			}
	return mirror_image;
}

Image custom(Image src_image, Matrix<double> kernel) {
	int radius_x = kernel.n_rows / 2, radius_y = kernel.n_cols / 2;
    Image m_img = get_mirror(src_image, radius_x, radius_y);
    Image dst_image(src_image.n_rows, src_image.n_cols);
    for (uint r = 0; r < dst_image.n_rows; r++)
    	for (uint c = 0; c < dst_image.n_cols; c++)
    	{
    		dpixel result(0, 0, 0);
    		for (int i = -radius_x; i <= radius_x; i++)
    			for (int j = -radius_y; j <= radius_y; j++)
    			result = sum(result, mul(m_img(radius_x + r + i, radius_y + c + j),
    									 kernel(i + radius_x, j + radius_y)));
    		double rr, gg, bb;
    		tie(rr, gg, bb) = result;
    		dst_image(r, c) = pixel(max(0., min(255., rr)), 
    							    max(0., min(255., gg)), 
    							    max(0., min(255., bb)));			
    	}
    return dst_image;
}

Image autocontrast(Image src_image, double fraction) {
    uint rows = src_image.n_rows, cols = src_image.n_cols;
    uint r = 0, g = 0, b = 0;
    float *hist = new float[256];
    for (int i = 0; i < 256; i++)
        hist[i] = 0;
    uMatrix gray = get_uMatrix(src_image);
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++)
            hist[gray(i, j)]++;

    uint y_min = 255 * fraction, y_max = 255 * (1 - fraction);
    float ymm = y_max - y_min;
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++)
        {
            tie(r, g, b) = src_image(i, j);
            if (r > y_min && r < y_max) r = (int(r) - int(y_min)) * 255. / ymm;
            else r = r <= y_min ? 0 : 255;
            if (g > y_min && g < y_max) g = (int(g) - int(y_min)) * 255. / ymm;
            else g = g <= y_min ? 0 : 255;
            if (b > y_min && b < y_max) b = (int(b) - int(y_min)) * 255. / ymm;
            else b = b <= y_min ? 0 : 255;
            src_image(i, j) = pixel(r, g, b);
        }

    return src_image;
}

Image gaussian(Image src_image, double sigma, int radius)  {
	uint size = 2 * radius + 1;
	dMatrix gauss(size, size);
	double sum = 0, koef = 1 / (sqrt(2 * PI) * sigma);
	for (int x = -radius; x <= radius; x++)
		for (int y = -radius; y <= radius; y++)
		{
			double res = koef * exp(-0.5 * (pow(x, 2) + pow(y, 2)) / pow(sigma, 2));
			sum += res;
			gauss(radius + x, radius + y) = res;
		}
	for (uint x = 0; x < size; x++)
		for (uint y = 0; y < size; y++)
			gauss(x, y) = gauss(x, y) / sum;
    return custom(src_image, gauss);
}

Image gaussian_separable(Image src_image, double sigma, int radius) {
	uint size = 2 * radius + 1;
	double *gauss = new double[size];
	double sum = 0, koef = 1 / (sqrt(2 * PI) * sigma);
	for (int i = -radius; i <= radius; i++)
	{
		double res = koef * exp(-0.5 * pow(i, i) / pow(sigma, 2));
		sum += res;
		gauss[radius + i] = res;
	}
	for (uint i = 0; i < size; i++)
		gauss[i] /= sum;
	dMatrix gauss_hor(size, 1), gauss_ver(1, size);
	for (uint i = 0; i < size; i++)
		gauss_hor(i, 0) = gauss_ver(0, i) = gauss[i];
	
    return custom(custom(src_image, gauss_hor), gauss_ver);
}

Image median(Image src_image, int radius) {
    uint rows = src_image.n_rows, cols = src_image.n_cols;
    Image dst_image(rows, cols);
    Image mir = get_mirror(src_image, radius, radius);
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++)
        {
            vector<int> R, G, B;
            for (int x = -radius; x <= radius; x++)
                for (int y = -radius; y <= radius; y++)
                {
                    int r, g, b;
                    tie(r, g, b) = mir(i + x + radius, j + y + radius);
                    R.push_back(r);
                    G.push_back(g);
                    B.push_back(b);
                }
            std::sort(R.begin(), R.end());
            std::sort(G.begin(), G.end());
            std::sort(B.begin(), B.end());
            int num = 2 * radius * (radius + 1) + 1;
            dst_image(i, j) = pixel(R[num], G[num], B[num]);
        }
    return dst_image;
}

Image median_linear(Image src_image, int radius) {

    return src_image;
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

void find_max(uint *mas, uint &start, uint end=1)
{
    uint mx = mas[start];
    for (uint i = start + 1; i < end; i++)
        if (mx < mas[i])
        {
            mx = mas[i];
            start = i;
        }
}

// delete borders on image
Image delete_borders(Image &src_image, uint th1=10, uint th2=20)
{
	uMatrix canny_matr = get_uMatrix(canny(src_image, th1, th2));
	uint rows = canny_matr.n_rows, cols = canny_matr.n_cols;
	uint border_x = float(rows) * 0.045f, border_y = float(cols) * 0.045f;

    //up
    uint *borders_pixels = new uint[border_x];
    for (uint x = 0; x < border_x; x++)
    {
        uint count = 0;
        for (uint y = 0; y < cols; y++)
            if (canny_matr(x, y))
                count++;
        borders_pixels[x] = count;
    }
    uint up_border = 0;
    find_max(borders_pixels, up_border, border_x);
    up_border += 4;
    find_max(borders_pixels, up_border, border_x);up_border += 4;

    //down
    borders_pixels = new uint[border_x];
    for (uint x = rows - 1; x >= rows - border_x; x--)
    {
        uint count = 0;
        for (uint y = 0; y < cols; y++)
            if (canny_matr(x, y))
                count++;
        borders_pixels[rows - x - 1] = count;
    }
    uint down_border = 0;
    find_max(borders_pixels, down_border, border_x);
    down_border += 4;
    find_max(borders_pixels, down_border, border_x);down_border += 4;
    down_border = rows - down_border - 1;

    //left
    borders_pixels = new uint[border_y];
    for (uint y = 0; y < border_y; y++)
    {
        uint count = 0;
        for (uint x = 0; x < rows; x++)
            if (canny_matr(x, y))
                count++;
        borders_pixels[y] = count;
    }
    uint left_border = 0;
    find_max(borders_pixels, left_border, border_y);
    left_border += 4;
    find_max(borders_pixels, left_border, border_y);left_border += 4;

    //right
    borders_pixels = new uint[border_y];
    for (uint y = cols - 1; y >= cols - border_y; y--)
    {
        uint count = 0;
        for (uint x = 0; x < rows; x++)
            if (canny_matr(x, y))
                count++;
        borders_pixels[cols - y - 1] = count;
    }
    uint right_border = 0;
    find_max(borders_pixels, right_border, border_y);
    right_border += 4;
    find_max(borders_pixels, right_border, border_y);right_border += 4;
    right_border = cols - right_border - 1;

	Image ans = src_image.submatrix(up_border, 
								    left_border, 
								    down_border - up_border, 
								    right_border - left_border);
	return ans;
}

Image canny(Image src_image, int threshold1, int threshold2) {
	Image gauss_image = gaussian_separable(src_image, 1.4, 2);
	uMatrix sobelx = get_uMatrix(sobel_x(gauss_image)), 
			sobely = get_uMatrix(sobel_y(gauss_image));
	uint rows = sobelx.n_rows, cols = sobelx.n_cols;
	dMatrix mod(rows, cols), theta(rows, cols);
	for (uint r = 0; r < rows; r++)
		for (uint c = 0; c < cols; c++)
		{
			mod(r, c) = sqrt(pow(sobelx(r, c), 2) + pow(sobely(r, c), 2));
			theta(r, c) = atan2(sobely(r, c), sobelx(r, c));
		}
	
	dMatrix new_mod = mod.deep_copy();
	for (uint r = 1; r < rows - 1; r++)
		for (uint c = 1; c < cols - 1; c++)
		{
			double angle1 = PI - theta(r, c);
			double angle2 = PI + angle1;
			if (angle2 >= 2 * PI) angle2 -= 2 * PI;
			
			int item1 = angle1 / (PI / 4), item2 = angle2 / (PI / 4);
			double i1 = get_item(new_mod, item1, r, c),
				   i2 = get_item(new_mod, item2, r, c);

			if (i1 > new_mod(r, c) || i2 > new_mod(r, c))
				new_mod(r, c) = 0;
		}
	
	uMatrix down_matrix(rows, cols), up_matrix(rows, cols);
	for (uint r = 0; r < rows; r++)
		for (uint c = 0; c < cols; c++)
		{
			double val = new_mod(r, c);
			down_matrix(r, c) = val > threshold1 ? 1 : 0;
			up_matrix(r, c) = val > threshold2 ? 1 : 0;
		}

	uMatrix map = labelling(down_matrix);
	
	set<uint> good_labels;
	
	for (uint i = 0; i < rows; i++)
		for (uint j = 0; j < cols; j++)
			if (up_matrix(i, j) > 0 && down_matrix(i, j) > 0)
				good_labels.insert(map(i, j));
	
	for (uint i = 0; i < rows; i++)
		for (uint j = 0; j < cols; j++)
			if (map(i, j) == 0 || good_labels.find(map(i, j)) == good_labels.end())
				map(i, j) = 0;
			else
				map(i, j) = 1;

    return to_Image(map, 255);
}

// returns labels of objects on image
uMatrix labelling(uMatrix matr)
{
	uint rows = matr.n_rows, cols = matr.n_cols;
	int km = 0, kn = 0, obj = 1;
	uint A, B, C;
	for (uint i = 0; i < rows; i++)
		for (uint j = 0; j < cols; j++)
		{
			kn = int(j) - 1;
			if (kn <= 0)
			{
				kn = 1;
				B = 0;
			}
			else
				B = matr(i, kn);
			km = int(i) - 1;
			if (km <= 0)
			{
				km = 1;
				C = 0;
			}
			else
				C = matr(km, j);
			A = matr(i, j);
			
			if (A == 0) {}
			else if (B == 0 && C == 0)
			{
				obj++;
				matr(i, j) = obj;
			}
			else if (B !=0 && C == 0)
				matr(i, j) = B;
			else if (B == 0 && C != 0)
				matr(i, j) = C;
			else if (B != 0 && C != 0)
			{
				matr(i, j) = B;
				if (B != C)
					for (uint ii = 0; ii < rows; ii++)
						for (uint jj = 0; jj < cols; jj++)
							if (matr(ii, jj) == C)
								matr(ii, jj) = B;
			}
		}
	return matr;
}

// returns neighbor number, for canny
double get_item(dMatrix &matr, int item, uint x_offset, uint y_offset)
{
	switch (item)
	{
		case 0: return matr(x_offset, y_offset + 1);
		case 1: return matr(x_offset - 1, y_offset + 1);
		case 2: return matr(x_offset - 1, y_offset);
		case 3: return matr(x_offset - 1, y_offset - 1);
		case 4: return matr(x_offset, y_offset - 1);
		case 5: return matr(x_offset + 1, y_offset - 1);
		case 6: return matr(x_offset + 1, y_offset);
		case 7: return matr(x_offset + 1, y_offset + 1);
		default: return 0;
	}
}