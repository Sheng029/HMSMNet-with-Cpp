#include "file.h"


// 获取文件下下所有文件的路径和文件名
void getFiles(const std::string path, std::vector<std::string>& files, std::vector<std::string>& names) {
	intptr_t h_file = 0;
	_finddata_t file_info;
	std::string p;

	if ((h_file = _findfirst(p.assign(path).append("/*").c_str(), &file_info)) != -1) {
		do {
			if ((file_info.attrib & _A_SUBDIR)) {
				if (strcmp(file_info.name, ".") != 0 && strcmp(file_info.name, "..") != 0)
					getFiles(p.assign(path).append("/").append(file_info.name), files, names);
			}
			else {
				files.push_back(p.assign(path).append("/").append(file_info.name));
				names.push_back(file_info.name);
			}
		} while (_findnext(h_file, &file_info) == 0);
		_findclose(h_file);
	}
}
