#include <iostream>
#include "EasyocrTensorRT.h"
#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds
#include "glob.h"

VizgardLogger::Logger *vizgardLogger = VizgardLogger::LoggerFactory::CreateConsoleLogger(VizgardLogger::INFO);

int main(int argc, char **argv)
{
    OCR::EasyOCR easyocr;
    easyocr.LoadEngine("/mnt/4B323B9107F693E2/TensorRT/OCR/easyocr-trt/weights/recognitionModel2.engine");

    std::string path = "/mnt/4B323B9107F693E2/TensorRT/OCR/EasyOCR/results/";
    glob::glob glob(path + "*.jpg");
    while (glob)
    {
        std::cout << "\n"
                  << glob.current_match() << std::endl;
        cv::Mat image_bgr = cv::imread(path + glob.current_match());
        std::string s = easyocr.EngineInference(image_bgr);
        std::cout << "[ RESULT ]:  " << glob.current_match() << "  \"" << s << "\"" << std::endl;
        // std::this_thread::sleep_for(std::chrono::seconds(3));

        glob.next();
    }
    // cv::Mat image_bgr = cv::imread("/mnt/4B323B9107F693E2/TensorRT/OCR/EasyOCR/results/In.jpg");
    // std::string s = easyocr.EngineInference(image_bgr);
    // std::cout << "[ RESULT ]:  \"" << s << "\"" << std::endl;
    return 0;
}