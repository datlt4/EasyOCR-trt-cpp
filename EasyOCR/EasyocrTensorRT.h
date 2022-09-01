#ifndef EASYOCR_TENSORRT_H
#define EASYOCR_TENSORRT_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cassert>
#include <tuple>
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "../TrtExec/Trtexec.h"

extern VizgardLogger::Logger *vizgardLogger;
namespace OCR
{
    const char c[] = {'\0', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '*' /*€*/, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    class EasyOCR : public TrtExec
    {
    protected:
        int batch_size = 1;

    public:
        EasyOCR() : TrtExec(){};
        ~EasyOCR(){};
        bool LoadEngine(const std::string &fileName);
        std::string EngineInference(cv::Mat &image);
        std::tuple<std::string, float> EngineInference2(cv::Mat &image);

    private:
        std::vector<float> prepareImage(cv::Mat &img);
        bool processInput(float *hostDataBuffer, const int batchSize, cudaStream_t &stream);
        std::tuple<std::vector<char>, float> postProcess(float *output, size_t length);
        IVizgardLogger iVLogger = IVizgardLogger();

        const int BATCH_SIZE = 1;
        const int IMAGE_WIDTH = -1;
        const int IMAGE_HEIGHT = 64;
        const int IMAGE_CHANNEL = 1;
        const int OUTPUT_LENGTH = -1;
        // const int OUTPUT_LENGTH_EXAMPLE = 89;
        const int OUTPUT_WIDTH = 97;
    };

    class CRAFT : public TrtExec
    {
    protected:
        int batch_size = 1;

    public:
        CRAFT() : TrtExec(){};
        ~CRAFT(){};

    private:
        std::vector<float> prepareImage(cv::Mat &img);
        std::vector<char> postProcess(float *output, size_t outSize);
        bool processInput(float *hostDataBuffer, const int batchSize, cudaStream_t &stream);

        IVizgardLogger iVLogger = IVizgardLogger();

        const int BATCH_SIZE = 1;
        const int IMAGE_WIDTH = -1;
        const int IMAGE_HEIGHT = 64;
        const int IMAGE_CHANNEL = 1;
        const int OUTPUT_LENGTH = -1;
        // const int OUTPUT_LENGTH_EXAMPLE = 89;
        const int OUTPUT_WIDTH = 97;
    };

    template <class T>
    void softmax(T *input, size_t size);

    std::tuple<std::vector<char>, float> decode_greedy(std::vector<std::tuple<unsigned int, float>> &preds_index);
}

#endif // EASYOCR_TENSORRT_H
