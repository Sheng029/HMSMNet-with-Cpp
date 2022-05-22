#pragma once

#include <io.h>
#include <vector>
#include <string>


// 获取指定文件夹下所有文件的路径和名字
void getFiles(const std::string path, std::vector<std::string>& files, std::vector<std::string>& names);
