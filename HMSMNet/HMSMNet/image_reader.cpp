#include "image_reader.h"


// 读取左图然后标准化，计算gx，gy，最后把它们的值放到left，gx，gy中
void readLeftImage(const std::string img_path, float* left, float* gx, float* gy) {
	cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	assert(img.cols = cols && img.rows == rows);

	// 计算均值
	float sum = 0.0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			sum += float(img.at<ushort>(i, j));
		}
	}
	float mean = sum / (rows * cols);

	// 计算标准差
	float sq_dev = 0.0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			sq_dev += float(img.at<ushort>(i, j) - mean) * float(img.at<ushort>(i, j) - mean);
		}
	}
	float std_dev = sqrt(sq_dev / (rows * cols));

	// 图像标准化
	cv::Mat nor_left = cv::Mat(rows, cols, CV_32F);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			left[i * cols + j] = (float(img.at<ushort>(i, j)) - mean) / std_dev;
			nor_left.at<float>(i, j) = (float(img.at<ushort>(i, j)) - mean) / std_dev;
		}
	}

	// 计算梯度gx，gy
	for (int i = 1; i < rows - 1; ++i) {
		for (int j = 1; j < cols - 1; ++j) {
			gx[i * cols + j] = nor_left.at<float>(i, j - 1) - nor_left.at<float>(i, j + 1);
			gy[i * cols + j] = nor_left.at<float>(i - 1, j) - nor_left.at<float>(i + 1, j);
		}
	}
	for (int i = 0; i < rows; ++i) {
		gx[i * cols + 0] = -1.0 * nor_left.at<float>(i, 1);
		gx[i * cols + cols - 1] = nor_left.at<float>(i, cols - 2);
	}
	for (int j = 0; j < cols; ++j) {
		gy[0 * cols + j] = -1.0 * nor_left.at<float>(1, j);
		gy[(rows - 1) * cols + j] = nor_left.at<float>(rows - 2, j);
	}
}


// 读取右图然后标准化，最后把值放到right中
void readRightImage(const std::string img_path, float* right) {
	cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	assert(img.cols = cols && img.rows == rows);

	// 计算均值
	float sum = 0.0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// data[i * cols + j] = float(img.at<ushort>(i, j));
			sum += float(img.at<ushort>(i, j));
		}
	}
	float mean = sum / (rows * cols);

	// 计算标准差
	float sq_dev = 0.0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			sq_dev += float(img.at<ushort>(i, j) - mean) * float(img.at<ushort>(i, j) - mean);
		}
	}
	float std_dev = sqrt(sq_dev / (rows * cols));

	// 图像标准化
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			right[i * cols + j] = (float(img.at<ushort>(i, j)) - mean) / std_dev;
		}
	}
}
