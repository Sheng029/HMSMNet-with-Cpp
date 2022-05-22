#pragma once

#include <tensorflow/c/c_api.h>
#include "image_reader.h"


class Network {
private:
	const char* tags = "serve";   // 默认
	const int num_inputs = 4;   // HMSMNet的输入是4个
	const int num_outputs = 1;   // 输出是1个
	TF_Graph* graph = nullptr;
	TF_Status* status = nullptr;
	TF_SessionOptions* session_opts = nullptr;
	TF_Session* session = nullptr;
	TF_Output* inputs = nullptr;
	TF_Output* outputs = nullptr;
	
public:
	Network(const std::string model_dir);
	~Network();
	void buildNet();   // 构建模型
	cv::Mat predict(const std::string left_path, const std::string right_path);   // 预测
};
