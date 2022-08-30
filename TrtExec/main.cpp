#include "Trtexec.h"

VizgardLogger::Logger *vizgardLogger = VizgardLogger::LoggerFactory::CreateConsoleLogger(VizgardLogger::INFO);

// ./Trtexec \
//     --onnx model.onnx \
//     --engine model.engine \
//     --inputName "input" \
//     --minShape 1x3x256x192 \
//     --optShape 8x3x256x192 \
//     --maxShape 32x3x256x192 \
//     --workspace 1024
//     --dynamicOnnx

int main(int argc, char **argv)
{
    OnnxParserConfig config;
    if (ParseCommandLine(argc, argv, config))
    {
        std::unique_ptr<TrtExec> executor = std::make_unique<TrtExec>(config);
        std::cout << config << std::endl;
        // if (config.dynamic)
        {
            executor->parseOnnxModel();
            executor->saveEngine(config.engine_dir);
        }
        VLOG(INFO) << "[ PASSED ]:\n"
                   << config << std::endl;
    }
    else
        VLOG(ERROR) << "[ ERROR ] STOP!!!" << std::endl;
}
