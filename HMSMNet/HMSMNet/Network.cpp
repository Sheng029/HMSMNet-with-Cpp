#include "Network.h"
#include "deallocator.h"


// ��ʼ��ģ��
Network::Network(const std::string model_dir) {
	// ���ļ����ж�ȡģ��
	graph = TF_NewGraph();
	status = TF_NewStatus();
	session_opts = TF_NewSessionOptions();
	session = TF_LoadSessionFromSavedModel(session_opts, nullptr, model_dir.c_str(), &tags, 1, graph, nullptr, status);
	assert(TF_GetCode(status) == TF_OK);
}


Network::~Network() {
	TF_DeleteGraph(graph);
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(session_opts);
	TF_DeleteStatus(status);
	delete[] inputs;
	inputs = nullptr;
	delete[] outputs;
	outputs = nullptr;
}


// ����ģ���������
void Network::buildNet() {
	// ģ�͵�����
	inputs = new TF_Output[num_inputs];
	TF_Output input_1 = { TF_GraphOperationByName(graph, "serving_default_input_1") };
	TF_Output input_2 = { TF_GraphOperationByName(graph, "serving_default_input_2") };
	TF_Output input_3 = { TF_GraphOperationByName(graph, "serving_default_input_3") };
	TF_Output input_4 = { TF_GraphOperationByName(graph, "serving_default_input_4") };
	assert(input_1.oper && input_2.oper && input_3.oper && input_4.oper);
	inputs[0] = input_1;   // left image
	inputs[1] = input_2;   // right image
	inputs[2] = input_3;   // gx
	inputs[3] = input_4;   // gy

	// ģ�͵����
	outputs = new TF_Output[num_outputs];
	TF_Output output = { TF_GraphOperationByName(graph, "StatefulPartitionedCall") };
	assert(output.oper);
	outputs[0] = output;
}


// Ԥ��
cv::Mat Network::predict(const std::string left_path, const std::string right_path) {
	// ģ�������ά��
	int64_t dims[] = { 1, rows, cols, 1 };
	int ndims = 4;
	size_t ndata = sizeof(float) * rows * cols;

	// ��ȡͼ��
	float* left = new float[rows * cols];
	float* right = new float[rows * cols];
	float* gx = new float[rows * cols];
	float* gy = new float[rows * cols];
	readLeftImage(left_path, left, gx, gy);
	readRightImage(right_path, right);

											 
	// ����ͼ����ͼ��gx��gy����ģ�͵�����
	TF_Tensor** input_values = new TF_Tensor*[num_inputs];
	input_values[0] = TF_NewTensor(TF_FLOAT, dims, ndims, left, ndata, &noOpDeallocator, 0);   // left image
	input_values[1] = TF_NewTensor(TF_FLOAT, dims, ndims, right, ndata, &noOpDeallocator, 0);;   // right image
	input_values[2] = TF_NewTensor(TF_FLOAT, dims, ndims, gx, ndata, &noOpDeallocator, 0);;   // gx
	input_values[3] = TF_NewTensor(TF_FLOAT, dims, ndims, gy, ndata, &noOpDeallocator, 0);;   // gy

	// Ԥ��
	TF_Tensor** output_values = new TF_Tensor*[num_outputs];
	TF_SessionRun(session, nullptr, inputs, input_values, num_inputs, outputs, output_values, num_outputs, nullptr, 0, nullptr, status);
	assert(TF_GetCode(status) == TF_OK);

	// �ͷ�
	delete[] input_values;
	input_values = nullptr;
	delete[] left;
	left = nullptr;
	delete[] right;
	right = nullptr;
	delete[] gx;
	gx = nullptr;
	delete[] gy;
	gy = nullptr;

	// ��Ԥ����ת����cv::Mat
	void* buff = TF_TensorData(output_values[0]);
	float* offsets = (float*)buff;
	cv::Mat disp = cv::Mat(rows, cols, CV_32F);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			disp.at<float>(i, j) = offsets[i * cols + j];
		}
	}
	delete[] output_values;
	output_values = nullptr;

	return disp;
}
