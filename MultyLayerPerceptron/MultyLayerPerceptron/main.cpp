
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "basic-includes.h"
#include "gnuplot-include.h"
#include "multy-layer-perceptron.h"


std::vector<MLP_DATA> LoadData(const std::string& folderPath) {
    std::vector<MLP_DATA> set;

    int l = -1;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            size_t label = (size_t)std::stoi(labelStr);

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd imgMat = Utils::ImageToMatrix(cv::imread(fullPathName));

            std::vector<double> input  =  Utils::FlatMatrix( imgMat );

            set.push_back({ input, label });

            if (label != l) { 
                l = label; 
                std::cout << "load data: [" << (label+1)*10 << "%]\n";
            }
        }
    }

    return set;
};

Eigen::MatrixXd TestingModelAccuracy(MLP* mlp, std::string path, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (const auto& entry : std::filesystem::directory_iterator(path.c_str())) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            int label = std::stoi(labelStr);

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));

            std::vector<double> inputs = Utils::FlatMatrix(input);

            std::vector<double> givenOutput = mlp->Classify(inputs);

            auto it = std::max_element(givenOutput.begin(), givenOutput.end());
            int givenLabel = std::distance(givenOutput.begin(), it);

            confusionMatrix(givenLabel, label) += 1.0;

            totalData++;

            if (givenLabel != label) { errors++; }
        }
    }

    (*accuracy) = 1.0 - ((double)errors/totalData);

    return confusionMatrix;
}

std::vector<double> EspectedVetorFromLabel(size_t l)
{ 
    auto label = std::vector<double>((size_t)10, 0.0 );
    label[l] = 1.0;
    return label;
}



int main(int argc, const char** argv)
{
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");


    std::vector<MLP_DATA> trainigDataSet  =  LoadData( "..\\..\\.resources\\train");


    MLP mlp  =  MlpBuilder()
                    .InputSize(28*28)
                    .Architecture({
                        LayerSignature(100, new Sigmoid(), 0.01),
                        LayerSignature(10, new Sigmoid(), 0.01, new CrossEntropy())
                    })
                    .MaxEpochs(20)
                    .ParseLabelToVector( EspectedVetorFromLabel )
                    .SaveOn("..\\..\\.resources\\gnuplot-output\\mlp\\mlp.json")
                    .Build();



    int ephocCounter = 0;
    mlp.Training(trainigDataSet, [&mlp, &trainigDataSet, &ephocCounter, &gnuplot](){
        double accuracy = 0.0;

        Eigen::MatrixXd confusionMatrix  =  TestingModelAccuracy(&mlp, "..\\..\\.resources\\train", &accuracy);

        std::cout << "Training Epoch: " << ephocCounter << "\n";
        std::cout << "Training Accuracy: " << accuracy << "\n\n";
        std::cout << confusionMatrix << "\n\n\n\n";

        gnuplot.out << ephocCounter << " " << accuracy << "\n";
        ephocCounter++;
    });


    gnuplot.out.close();
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01","1.05");
    gnuplot.Grid("1", "0.1");
    gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l title \"Training Accuracy\" \n";
    gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
    gnuplot << " \n";


    double accuracy = 0.0;
    Eigen::MatrixXd confusionMatrix  =  TestingModelAccuracy(&mlp, "..\\..\\.resources\\test", &accuracy);
    std::cout << "ending training:\n";
    std::cout << "Testing Accuracy: " << accuracy << "\n\n";
    std::cout << confusionMatrix << "\n\n\n\n";


    std::cout << "\n\n[SUCESSO]";

    return 0;
}

