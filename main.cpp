#include <iostream>
#include "EasyocrTensorRT.h"
#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds
#include "c_glob.h"

VizgardLogger::Logger *vizgardLogger = VizgardLogger::LoggerFactory::CreateConsoleLogger(VizgardLogger::INFO);

int main(int argc, char **argv)
{
    OCR::EasyOCR easyocr;
    easyocr.LoadEngine("../weights/recognitionModel.engine");

    std::string path = "../samples/extracted/";
    c_glob::glob _glob(path + "*.jpg");
    while (_glob)
    {
        std::cout << "\n"
                  << _glob.current_match() << std::endl;
        cv::Mat image_bgr = cv::imread(path + _glob.current_match());

        std::string s = easyocr.EngineInference(image_bgr);
        std::cout << "[ RESULT ]:  " << _glob.current_match() << "  \"" << s << "\"" << std::endl;

        std::tuple<std::string, float> r = easyocr.EngineInference2(image_bgr);
        std::cout << "[ RESULT2 ]:  " << _glob.current_match() << "  \""
                  << std::get<0>(r) << "\""
                  << "  " << std::get<1>(r) << std::endl;

        // std::this_thread::sleep_for(std::chrono::seconds(3));

        _glob.next();
    }
    // cv::Mat image_bgr = cv::imread("/mnt/4B323B9107F693E2/TensorRT/OCR/EasyOCR/results/In.jpg");
    // std::string s = easyocr.EngineInference(image_bgr);
    // std::cout << "[ RESULT ]:  \"" << s << "\"" << std::endl;
    return 0;
}