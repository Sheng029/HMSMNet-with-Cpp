#include <iostream>
#include <iomanip>
#include "Network.h"
#include "deallocator.h"
#include "file.h"


int main() {
	// 模型、图像、预测输出文件夹路径
	const std::string model_dir = "E:/HMSMNet with CPP/testing/pb_model";
	const std::string left_dir = "E:/HMSMNet with CPP/testing/data/left";
	const std::string right_dir = "E:/HMSMNet with CPP/testing/data/right";
	const std::string pred_dir = "E:/HMSMNet with CPP/testing/data/pred";

	// 构建网络
	Network net = Network(model_dir);
	net.buildNet();

	// 批量预测
	std::vector<std::string> left_files, right_files;
	std::vector<std::string> left_names, right_names;
	getFiles(left_dir, left_files, left_names);
	getFiles(right_dir, right_files, right_names);
	assert(left_files.size() == right_files.size());
	assert(left_names.size() == right_names.size());
	assert(left_files.size() == left_names.size());

	for (int i = 0; i < left_files.size(); ++i) {
		cv::Mat disp = net.predict(left_files[i], right_files[i]);
		int pos = left_names[i].find("left");
		std::string saved_name = left_names[i].replace(pos, 4, "disparity");
		std::string saved_dir = pred_dir;
		std::string saved_path = saved_dir.append("/").append(saved_name);
		cv::imwrite(saved_path, disp);
	}

	return 0;
}
