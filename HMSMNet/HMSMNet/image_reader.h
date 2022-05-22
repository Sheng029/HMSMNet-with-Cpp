#pragma once

#include <string>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "net_config.h"


// ��ȡ��ͼȻ���׼��������gx��gy���������ǵ�ֵ�ŵ�left��gx��gy��
void readLeftImage(const std::string img_path, float* left, float* gx, float* gy);


// ��ȡ��ͼȻ���׼��������ֵ�ŵ�right��
void readRightImage(const std::string img_path, float* right);
