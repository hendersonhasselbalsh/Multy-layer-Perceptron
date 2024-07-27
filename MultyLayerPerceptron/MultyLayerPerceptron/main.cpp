#include <iostream> 
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) 
{ 
    cv::Mat image = cv::imread("..\\..\\.resources\\test\\0_3.jpg", cv::IMREAD_GRAYSCALE); 
    cv::imshow("Window Name", image); 

    std::cout << "[SUCESSO]\n"; 

    cv::waitKey(0); 
    return 0; 
}