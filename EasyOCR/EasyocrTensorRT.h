#ifndef YOLOV4_H
#define YOLOV4_H

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
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "../vizgard/common.h"
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

    std::vector<char> decode_greedy(std::vector<unsigned int> preds_index);
}

#endif // YOLOV4_H
