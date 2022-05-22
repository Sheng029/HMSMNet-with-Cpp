#pragma once

#include <tensorflow/c/c_api.h>
#include "image_reader.h"


class Network {
private:
	const char* tags = "serve";   // Ĭ��
	const int num_inputs = 4;   // HMSMNet��������4��
	const int num_outputs = 1;   // �����1��
	TF_Graph* graph = nullptr;
	TF_Status* status = nullptr;
	TF_SessionOptions* session_opts = nullptr;
	TF_Session* session = nullptr;
	TF_Output* inputs = nullptr;
	TF_Output* outputs = nullptr;
	
public:
	Network(const std::string model_dir);
	~Network();
	void buildNet();   // ����ģ��
	cv::Mat predict(const std::string left_path, const std::string right_path);   // Ԥ��
};
