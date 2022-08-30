#include "EasyocrTensorRT.h"
using namespace VizgardTrt;

bool OCR::EasyOCR::LoadEngine(const std::string &fileName)
{
    bool r = this->loadEngine(fileName);
    assert(r);
    return r;
}

std::vector<float> OCR::EasyOCR::prepareImage(cv::Mat &img)
{
    cv::Mat img_cv_grey;
    if (img.channels() == 3)
    {
        cv::cvtColor(img, img_cv_grey, cv::COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4)
    {
        cv::cvtColor(img, img_cv_grey, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        img.copyTo(img_cv_grey);
    }

    int width, max_width;
    float ratio = img_cv_grey.cols / (float)img_cv_grey.rows;

    // std::cout << TAGLINE << "ratio: " << ratio << std::endl;
    cv::Mat rsz_img;
    float max_ratio = (ratio > 1.0f) ? std::ceil(ratio) : 1.0f;
    width = IMAGE_HEIGHT * ratio;
    max_width = IMAGE_HEIGHT * max_ratio;

    cv::resize(img_cv_grey, rsz_img, cv::Size(width, IMAGE_HEIGHT), cv::INTER_CUBIC);
    std::vector<float> result(long(BATCH_SIZE * max_width * IMAGE_HEIGHT * IMAGE_CHANNEL));
    float *data = result.data();
    if (!img_cv_grey.data)
        return result;

    cv::Mat flt_img = cv::Mat(cv::Size(max_width, IMAGE_HEIGHT), CV_32FC1, 0.f);
    rsz_img.convertTo(rsz_img, CV_32FC1, 2.0 / 255, -1.0f);
    for (int row_num = 0; row_num < IMAGE_HEIGHT; ++row_num)
    {
        float *flt_img_row = flt_img.ptr<float>(row_num);
        float *rsz_img_row = rsz_img.ptr<float>(row_num);
        float C = *(rsz_img_row + width - 1);
        std::fill(flt_img_row + width, flt_img_row + max_width, C);
    }
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));

    // HWC TO CHW
    int channelLength = max_width * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
        cv::Mat(IMAGE_HEIGHT, max_width, CV_32FC1, data)};
    cv::split(flt_img, split_img);

    return result;
}

bool OCR::EasyOCR::processInput(float *hostDataBuffer, const int widthImg, cudaStream_t &stream)
{
    // std::vector< void* > input_buffers(this->prediction_engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < this->prediction_engine->getNbBindings(); ++i)
    {
        int32_t binding_size = volume(this->prediction_engine->getBindingDimensions(i)) * widthImg * sizeof(float);
        binding_size = (binding_size > 0) ? binding_size : -binding_size;
        // std::cout << "Size of: " << binding_size << std::endl;
        if (this->prediction_engine->bindingIsInput(i))
        {
            input_buffers.emplace_back(new float());
            cudaMalloc(&input_buffers.back(), binding_size);
            prediction_input_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
        else
        {
            output_buffers.emplace_back(new float());
            cudaMalloc(&output_buffers.back(), binding_size);
            prediction_output_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
    }

    if (prediction_input_dims.empty() || prediction_output_dims.empty())
    {
        VLOG(ERROR) << "Expect at least one input and one output for network";
        return false;
    }

    float *gpu_input_0 = (float *)input_buffers[0];
    // TensorRT copy way
    // Host memory for input buffer
    if (cudaMemcpyAsync(gpu_input_0, hostDataBuffer, size_t(BATCH_SIZE * IMAGE_HEIGHT * widthImg * IMAGE_CHANNEL * sizeof(float)), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        VLOG(ERROR) << "Input corrupted or CUDA error, abort!";
        return false;
    }
    return true;
}

std::string OCR::EasyOCR::EngineInference(cv::Mat &image)
{
    std::string text{""};
    std::vector<float> curInput = prepareImage(image);
    if (!curInput.data())
    {
        return text;
    }

    this->processInput(curInput.data(), curInput.size() / 64, stream);

    std::vector<void *> predicitonBindings = {(float *)input_buffers[0], (float *)output_buffers[0]};
    // VLOG(INFO) << "Input " << log_cuda_bf(curInput.size(), (void *)input_buffers[0], 100);
    this->prediction_context->setBindingDimensions(0, nvinfer1::Dims4(BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, curInput.size() / 64));
    this->prediction_context->enqueue(BATCH_SIZE, predicitonBindings.data(), 0, nullptr);
    // VLOG(INFO) << "Output: " << log_cuda_bf(1000, predicitonBindings[1], 100);
    int output_length = static_cast<int>(curInput.size() / 64.0f / 4.0f) - 1;
    std::cout << "[ output_length ]: " << output_length << std::endl;
    std::vector<float> output(BATCH_SIZE * output_length * OUTPUT_WIDTH);
    cudaMemcpy(output.data(), predicitonBindings[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<char> list_char = postProcess(output.data(), output_length);
    text = std::string(list_char.data());
    clearBuffer();
    return text;
}

std::vector<char> OCR::EasyOCR::postProcess(float *output, size_t length)
{
    float *out = output;
    cv::Mat result_matrix = cv::Mat(length, OUTPUT_WIDTH, CV_32FC1, out);
    std::vector<unsigned int> preds_index;
    for (int row_num = 0; row_num < length; row_num++)
    {
        float *row = result_matrix.ptr<float>(row_num);
        softmax(row, OUTPUT_WIDTH);
        float *max_pos = std::max_element(row, row + OUTPUT_WIDTH);
        unsigned int pred_index = static_cast<unsigned int>(max_pos - row);
        preds_index.push_back(pred_index);
    }
    // std::vector<char> list_char(length);
    std::vector<char> list_char = decode_greedy(preds_index);
    return list_char;
}

template <class T>
void OCR::softmax(T *input, size_t size)
{
    assert(0 <= size <= sizeof(input) / sizeof(T));

    int i;
    T m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < size; ++i)
        if (m < input[i])
            m = input[i];

    sum = 0.0;
    for (i = 0; i < size; ++i)
        sum += std::exp(input[i] - m);

    constant = m + std::log(sum);
    sum = 0.0;
    for (i = 0; i < size; ++i)
        input[i] = std::exp(input[i] - constant);

    sum = std::accumulate(input, input + size, static_cast<T>(0), std::plus<T>());
    for (i = 0; i < size; ++i)
        input[i] /= sum;
}

std::vector<char> OCR::decode_greedy(std::vector<unsigned int> preds_index)
{
    std::vector<char> list_char;
    for (int i = 0; i < preds_index.size(); ++i)
    {
        bool c = ((i == 0) ? true : (preds_index[i - 1]) != preds_index[i]) & (preds_index[i] != 0);
        if (c)
        {
            list_char.push_back(OCR::c[preds_index[i]]);
        }
    }
    list_char.push_back('\0');
    return list_char;
}
