#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tie;
using std::sqrt;
using std::atan2;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

typedef Matrix<float> fMatrix;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}


/////////////////// CODE STARTS HERE //////////////////////

#ifndef PI
#define PI 3.141592654
#endif

class HorizontalSobel
{
public:
    static const uint vert_radius = 1, hor_radius = 3;
    float operator()(const fMatrix &matr) const
    { return matr(0, 0) - matr(0, 2); }
};

class VerticalSobel
{
public:
    static const uint vert_radius = 3, hor_radius = 1;
    float operator()(const fMatrix &matr) const
    { return matr(0, 0) - matr(2, 0); }
};

void normalize(vector<float> &vec)
{
    float sum = 0;  
    for (uint h = 0; h < vec.size(); h++)
        sum += vec[h] * vec[h];
    sum = sum > 0 ? sqrt(sum) : 1;
    for (uint h = 0; h < vec.size(); h++)
        vec[h] /= sum;
}

BMP* submatrix(BMP *img, int w0, int h0, int width, int height)
{
    BMP* smatr = new BMP();
    smatr->SetSize(width, height);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            smatr->SetPixel(i, j, img->GetPixel(w0 + i, h0 + j));
    return smatr;
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {

        // Get image & consts
		BMP* img = data_set[image_idx].first;
		int height = img->TellHeight(), width = img->TellWidth();
		int hog_img_blocks = 8, dir_blocks = 24, lbp_img_blocks = 4, rgb_img_blocks = 8;
		int hog_block_width = width / hog_img_blocks, hog_block_height = height / hog_img_blocks;
		int lbp_block_width = width / lbp_img_blocks, lbp_block_height = height / lbp_img_blocks;
		int rgb_block_width = width / rgb_img_blocks, rgb_block_height = height / rgb_img_blocks;		
		vector<float> image_features;
		
		// 1. Get gray image
		fMatrix gray(width, height);
		for (int i = 0; i < width; i++)
		    for (int j = 0; j < height; j++)
		    {
		        RGBApixel p = img->GetPixel(i, j);
		        gray(i, j) = 0.299f * p.Red + 0.587f * p.Green + 0.114f * p.Blue;
		    }

        // 2. Get gradients
        fMatrix sobel_hor = gray.unary_map(HorizontalSobel()),
                sobel_vert = gray.unary_map(VerticalSobel());

        // 3. Get module & direction
        fMatrix mod(width, height), dir(width, height);
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
            {
                float x = sobel_hor(i, j), y = sobel_vert(i, j);
                mod(i, j) = sqrt(x * x + y * y);
                dir(i, j) = atan2(y, x);
            }
        
        // 4. Get histograms
        for (int i = 0; i < hog_img_blocks; i++)
            for (int j = 0; j < hog_img_blocks; j++)
            {
                vector<float> hist;
                hist.resize(dir_blocks);
                for (int h = 0; h < dir_blocks; h++)
                    hist[h] = 0;
                    
                fMatrix mod_block = mod.submatrix(i * hog_block_width, j * hog_block_height, 
                                                  hog_block_width, hog_block_height), 
                        dir_block = dir.submatrix(i * hog_block_width, j * hog_block_height, 
                                                  hog_block_width, hog_block_height);
                                                  
                for (int bw = 0; bw < hog_block_width; bw++)
                    for (int bh = 0; bh < hog_block_height; bh++)
                    {
                        int hist_segment = (PI + dir_block(bw, bh)) / (2 * PI / dir_blocks);
                        hist[hist_segment] += mod_block(bw, bh);
                    }
                 
                // 5. Normalize histogram
                normalize(hist);
                
                //6. Concatenate
                image_features.insert(image_features.end(), hist.begin(), hist.end());
            }
        
        // DOP1. LBP
        for (int i = 0; i < lbp_img_blocks; i++)
            for (int j = 0; j < lbp_img_blocks; j++)
            {
                vector<float> hist;
                hist.resize(256);
                for (int h = 0; h < 256; h++)
                    hist[h] = 0;
                    
                fMatrix block = gray.submatrix(i * lbp_block_width, j * lbp_block_height, 
                                              lbp_block_width, lbp_block_height);
                                              
                for (int w = 1; w < lbp_block_width - 1; w++)
                    for (int h = 1; h < lbp_block_height - 1; h++)
                    {
                        float val = block(w, h);
                        int count = 0, sum = 0;
                        for (int w_offset = -1; w_offset <= -1; w_offset++)
                            for (int h_offset = -1; h_offset <= -1; h_offset++)
                                if (w_offset == 0 && h_offset == 0)
                                    continue;
                                else
                                    sum += (val >= block(w + w_offset, h + h_offset) ? 0 : 1) * 
                                           pow(2, count++); 
                    }
                
                normalize(hist);
                image_features.insert(image_features.end(), hist.begin(), hist.end());                
            }
        
        // DOP2. RGB
        float rgb_block_square = rgb_block_width * rgb_block_height;
        for (int i = 0; i < rgb_img_blocks; i++)
            for (int j = 0; j < rgb_img_blocks; j++)
            {
                BMP* block = submatrix(img, i * rgb_block_width, j * rgb_block_height, 
                                           rgb_block_width, rgb_block_height);
                float r = 0, g = 0, b = 0;
                for (int w = 0; w < rgb_block_width; w++)
                    for (int h = 0; h < rgb_block_height; h++)
                    {
                        RGBApixel p = block->GetPixel(w, h);
                        r += p.Red;
                        g += p.Green;
                        b += p.Blue;
                    }
               image_features.push_back(r / rgb_block_square / 255.f);
               image_features.push_back(g / rgb_block_square / 255.f);
               image_features.push_back(b / rgb_block_square / 255.f);
            }

        
        // Concatenate with final deskriptor
        features->push_back(make_pair(image_features, data_set[image_idx].second));

    }
}

/////////////////// CODE ENDS HERE ////////////////////////

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.1;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
