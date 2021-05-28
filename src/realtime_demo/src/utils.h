#pragma once
#include <string>
#include <opencv2/opencv.hpp>
enum SystemMode { VO = 0, VIO };
struct Parameters {
  SystemMode slam_mode_;
  int start_image_index_;
  std::string img_path_;
  std::string imu_path_;
  std::string pose_log_path_;
  
};

class Config {
public:
  explicit Config(const std::string &config_file_path);
  //friend std::ostream &operator<<(std::ostream &os, const Config &cfg);

  const std::string config_file_path_;

  Parameters parameters_;
};