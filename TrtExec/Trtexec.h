#ifndef TRT_EXEC_H
#define TRT_EXEC_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <assert.h>
#include <map>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <map>
#include <numeric>
#include <iomanip>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "fstream"
#include "VizgardLogger.h"

#ifndef TAGLINE
#define TAGLINE "\t<L" << __LINE__ << "> "
#endif // TAGLINE

struct OnnxParserConfig
{
    int minBatchSize;
    int minImageChannel;
    int minImageHeight;
    int minImageWidth;
    int optBatchSize;
    int optImageChannel;
    int optImageHeight;
    int optImageWidth;
    int maxBatchSize;
    int maxImageChannel;
    int maxImageHeight;
    int maxImageWidth;
    int workspace{1ULL << 30};
    std::string inputName;
    std::string onnx_dir;
    std::string engine_dir;
    bool dynamicOnnx{false};
    friend std::ostream &operator<<(std::ostream &os, const OnnxParserConfig config)
    {
        os << "  --onnx         : " << config.onnx_dir << std::endl
           << "  --engine       : " << config.engine_dir << std::endl
           << "  --minShape     : " << config.minBatchSize << "x" << config.minImageChannel << "x" << config.minImageHeight << "x" << config.minImageWidth << std::endl
           << "  --optShape     : " << config.optBatchSize << "x" << config.optImageChannel << "x" << config.optImageHeight << "x" << config.optImageWidth << std::endl
           << "  --maxShape     : " << config.maxBatchSize << "x" << config.maxImageChannel << "x" << config.maxImageHeight << "x" << config.maxImageWidth << std::endl
           << "  --dynamicOnnx  : " << (config.dynamicOnnx ? "True" : "False") << std::endl;
        return os;
    }
};

void ShowHelpAndExit(const char *szBadOption);
bool ParseCommandLine(int argc, char *argv[], OnnxParserConfig &config);

extern VizgardLogger::Logger *vizgardLogger;

using Severity = nvinfer1::ILogger::Severity;

struct VizgardDestroyPtr
{
    template <class T>
    void operator()(T *obj) const
    {
        if (obj != nullptr)
        {
            obj->destroy();
        }
    }
};

template <class T>
using VizgardUniquePtr = std::unique_ptr<T, VizgardDestroyPtr>;

template <typename T>
VizgardUniquePtr<T> makeUnique(T *t)
{
    return VizgardUniquePtr<T>{t};
}

struct VizgardOnnxParser
{
    VizgardUniquePtr<nvonnxparser::IParser> onnxParser;
    operator bool() const
    {
        return !!(onnxParser);
    }
};

class IVizgardLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        static VizgardLogger::LogLevel map[] = {
            VizgardLogger::FATAL, VizgardLogger::ERROR, VizgardLogger::WARNING, VizgardLogger::INFO, VizgardLogger::TRACE};
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {
            VizgardLogger::LogTransaction(vizgardLogger, map[(int)severity], __FILE__, __LINE__, __FUNCTION__).GetStream() << msg;
        }
    }
    nvinfer1::ILogger &getTRTLogger()
    {
        return *this;
    }
};

class TrtExec
{
protected:
    VizgardOnnxParser onnxParser;
    VizgardUniquePtr<nvinfer1::INetworkDefinition> prediction_network;
    VizgardUniquePtr<nvinfer1::ICudaEngine> prediction_engine{nullptr};
    VizgardUniquePtr<nvinfer1::IExecutionContext> prediction_context{nullptr};

    IVizgardLogger iVLogger = IVizgardLogger();
    int batch_size = 1;
    std::vector<nvinfer1::Dims> prediction_input_dims;
    std::vector<nvinfer1::Dims> prediction_output_dims;

    std::vector<void *> input_buffers; // buffers for input and output data
    std::vector<void *> output_buffers;

    cudaStream_t stream;
    int maxBatchSize;

    int32_t getNbBindings();
    nvinfer1::Dims getBindingDimensions(int32_t bindingIndex);
    nvinfer1::DataType getBindingDataType(int32_t bindingIndex);
    int getMaxBatchSize();
    bool clearBuffer(bool freeInput = true, bool freeOutput = true);

public:
    TrtExec(const OnnxParserConfig &info) : info{info}
    {
        cudaStreamCreate(&stream);
    }
    TrtExec() { cudaStreamCreate(&stream); }
    ~TrtExec()
    {
        cudaStreamDestroy(stream);
        for (void *buf : output_buffers)
            cudaFree(buf);
        for (void *buf : input_buffers)
            cudaFree(buf);
        this->prediction_context.reset();
        this->prediction_engine.reset();
        this->prediction_network.reset();
        this->onnxParser.onnxParser.reset();
    }

    /*virtual*/ bool parseOnnxModel();
    /*virtual*/ bool saveEngine(const std::string &fileName);
    /*virtual*/ bool loadEngine(const std::string &fileName);

private:
    OnnxParserConfig info;
};

namespace VizgardTrt
{
    inline int64_t volume(const nvinfer1::Dims &d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    inline std::string log_cuda_bf(nvinfer1::Dims const &dim_shape, void *cuda_buffer, int number_p)
    {
        std::ostringstream oss;
        if (!cuda_buffer)
            oss << "Null cuda buffer !" << std::endl;
        oss << "Buffer size: ";
        for (size_t i = 0; i < dim_shape.nbDims - 1; ++i)
            oss << dim_shape.d[i] << "x";
        oss << dim_shape.d[dim_shape.nbDims - 1] << ".  Some elements: ";
        int64_t v = volume(dim_shape);
        std::vector<float> cpu_output(v > 0 ? v : -v);
        cudaMemcpy(cpu_output.data(), (float *)cuda_buffer, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < number_p; i++)
            oss << cpu_output[i] << " ";
        oss << std::endl;
        return oss.str();
    }

    inline std::string log_cuda_bf(size_t len, void *cuda_buffer, int number_p)
    {
        std::ostringstream oss;
        if (!cuda_buffer)
            oss << "Null buffer !" << std::endl;
        oss << "Buffer size: ";
        oss << "[ " << len << " ]"
            << ".  Some elements: ";
        std::vector<float> cpu_output(len);
        cudaMemcpy(cpu_output.data(), (float *)cuda_buffer, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < number_p; i++)
            oss << cpu_output[i] << " ";
        oss << std::endl;
        return oss.str();
    }

    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
            return 1;
        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }
}
#endif // TRT_EXEC_H