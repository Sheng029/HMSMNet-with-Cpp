#pragma once

#include <string>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "net_config.h"


// 读取左图然后标准化，计算gx，gy，最后把它们的值放到left，gx，gy中
void readLeftImage(const std::string img_path, float* left, float* gx, float* gy);


// 读取右图然后标准化，最后把值放到right中
void readRightImage(const std::string img_path, float* right);
