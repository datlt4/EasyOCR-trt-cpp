#include "Trtexec.h"

bool TrtExec::parseOnnxModel()
{
    // const char inputName[10] = "input";
    VizgardUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(iVLogger)};
    // We need to define explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    VizgardUniquePtr<nvinfer1::INetworkDefinition> prediction_network{builder->createNetworkV2(explicitBatch)};
    // VizgardUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    VizgardUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*prediction_network, iVLogger)};
    // parse ONNX
    if (!parser->parseFromFile(info.onnx_dir.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        VLOG(ERROR) << "ERROR: could not parse the model.";
        return false;
    }
    VizgardUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        VLOG(ERROR) << "Create builder config failed.";
        return false;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(info.workspace);
    if (info.dynamicOnnx)
    {
        builder->setMaxBatchSize(info.maxBatchSize);
        // generate TensorRT engine optimized for the target platform
        nvinfer1::IOptimizationProfile *profileCalib = builder->createOptimizationProfile();
        // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
        profileCalib->setDimensions(info.inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{info.minBatchSize, info.minImageChannel, info.minImageHeight, info.minImageWidth});
        profileCalib->setDimensions(info.inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{info.optBatchSize, info.optImageChannel, info.optImageHeight, info.optImageWidth});
        profileCalib->setDimensions(info.inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{info.maxBatchSize, info.maxImageChannel, info.maxImageHeight, info.maxImageWidth});
        config->addOptimizationProfile(profileCalib);
    }
    else
    {
        builder->setMaxBatchSize(1);
    }
    this->prediction_engine.reset(builder->buildEngineWithConfig(*prediction_network, *config));
    this->prediction_context.reset(this->prediction_engine->createExecutionContext());
    return true;
}

bool TrtExec::saveEngine(const std::string &fileName)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        VLOG(ERROR) << "Cannot open engine file: " << fileName;
        return false;
    }
    VizgardUniquePtr<nvinfer1::IHostMemory> serializedEngine{this->prediction_engine->serialize()};
    if (serializedEngine == nullptr)
    {
        VLOG(ERROR) << "Engine serialization failed";
        return false;
    }
    engineFile.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

bool TrtExec::loadEngine(const std::string &fileName)
{
    std::ifstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        VLOG(ERROR) << "Cannot open engine file: " << fileName;
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if (!engineFile.good())
    {
        VLOG(ERROR) << "Error loading engine file";
        return false;
    }

    VizgardUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(iVLogger.getTRTLogger())};
    this->prediction_engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    this->prediction_context.reset(this->prediction_engine->createExecutionContext());
    this->maxBatchSize = this->prediction_engine->getMaxBatchSize();
    return this->prediction_engine != nullptr;
}

int32_t TrtExec::getNbBindings()
{
    return this->prediction_engine->getNbBindings();
}

nvinfer1::Dims TrtExec::getBindingDimensions(int32_t bindingIndex)
{
    return this->prediction_engine->getBindingDimensions(bindingIndex);
}

nvinfer1::DataType TrtExec::getBindingDataType(int32_t bindingIndex)
{
    return this->prediction_engine->getBindingDataType(bindingIndex);
}

int TrtExec::getMaxBatchSize()
{
    return this->prediction_engine->getMaxBatchSize();
}

bool TrtExec::clearBuffer(bool freeInput, bool freeOutput)
{
    this->prediction_input_dims.clear();
    this->prediction_output_dims.clear();
    try
    {
        if (freeInput)
            for (void *buf : input_buffers)
                cudaFree(buf);

        if (freeOutput)
            for (void *buf : output_buffers)
                cudaFree(buf);
    }
    catch (std::runtime_error &e)
    {
        VLOG(ERROR) << e.what() << std::endl;
        return false;
    }
    input_buffers.clear();
    output_buffers.clear();
    // TODO: Properly re wrote this
    return true;
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "    --onnx [PATH]       : path to Onnx file" << std::endl
        << "    --engine [PATH]     : name of output Engine file" << std::endl
        << "    --dynamicOnnx       : indicate that build engine with Dynamic Batch Size" << std::endl
        << "    --minShape [BxCxHxW]: min input shape" << std::endl
        << "    --optShape [BxCxHxW]: optimization input shape" << std::endl
        << "    --maxShape [BxCxHxW]: max input shape" << std::endl
        << "    --workspace [Int]   : max workspace size in MB" << std::endl;

    oss << std::endl;

    if (bThrowError)
        throw std::invalid_argument(oss.str());
    else
        std::cout << oss.str();
}

bool ParseCommandLine(int argc, char *argv[], OnnxParserConfig &config)
{
    if (argc <= 1)
    {
        ShowHelpAndExit();
        return false;
    }
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == std::string("--help"))
        {
            ShowHelpAndExit();
            return false;
        }
        else if (std::string(argv[i]) == std::string("--onnx"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--onnx");
                return false;
            }

            else
                config.onnx_dir = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--engine"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--engine");
                return false;
            }
            else
                config.engine_dir = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--inputName"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--inputName");
                return false;
            }
            else
                config.inputName = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--dynamicOnnx"))
        {
            config.dynamicOnnx = true;
        }
        else if (std::string(argv[i]) == std::string("--minShape"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--minShape");
                return false;
            }
            else
            {
                std::stringstream minShape{argv[i]};
                std::vector<std::string> result;
                std::string item;
                while (getline(minShape, item, 'x'))
                {
                    result.push_back(item);
                }
                if (result.size() != 4)
                {
                    ShowHelpAndExit("--minShape");
                    return false;
                }
                config.minBatchSize = std::atoi(result.at(0).c_str());
                config.minImageChannel = std::atoi(result.at(1).c_str());
                config.minImageHeight = std::atoi(result.at(2).c_str());
                config.minImageWidth = std::atoi(result.at(3).c_str());
            }
            continue;
        }
        else if (std::string(argv[i]) == std::string("--optShape"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--optShape");
                return false;
            }
            else
            {
                std::stringstream optShape{argv[i]};
                std::vector<std::string> result;
                std::string item;
                while (getline(optShape, item, 'x'))
                    result.push_back(item);

                if (result.size() != 4)
                {
                    ShowHelpAndExit("--optShape");
                    return false;
                }
                config.optBatchSize = std::atoi(result.at(0).c_str());
                config.optImageChannel = std::atoi(result.at(1).c_str());
                config.optImageHeight = std::atoi(result.at(2).c_str());
                config.optImageWidth = std::atoi(result.at(3).c_str());
            }
            continue;
        }
        else if (std::string(argv[i]) == std::string("--maxShape"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--maxShape");
                return false;
            }
            else
            {
                std::stringstream maxShape{argv[i]};
                std::vector<std::string> result;
                std::string item;
                while (getline(maxShape, item, 'x'))
                    result.push_back(item);

                if (result.size() != 4)
                {
                    ShowHelpAndExit("--maxShape");
                    return false;
                }
                config.maxBatchSize = std::atoi(result.at(0).c_str());
                config.maxImageChannel = std::atoi(result.at(1).c_str());
                config.maxImageHeight = std::atoi(result.at(2).c_str());
                config.maxImageWidth = std::atoi(result.at(3).c_str());
            }
            continue;
        }
        else if (std::string(argv[i]) == std::string("--workspace"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--workspace");
                return false;
            }
            else
                config.workspace = std::stoi(argv[i]) * (1ULL << 20);
            continue;
        }
        else
        {
            {
                ShowHelpAndExit((std::string("input not include ") + std::string(argv[i])).c_str());
                return false;
            }
        }
    }
    return true;
}
