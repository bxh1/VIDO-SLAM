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
  explicit Config(const std::string &config_file_path){
     cv::FileStorage fsSettings(config_file_path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image_path"] >> parameters_.img_path_;
    fsSettings["imu_path"] >> parameters_.imu_path_;
    parameters_.start_image_index_ = static_cast<int>(fsSettings["start_index"]);
    parameters_.slam_mode_ = ((static_cast<int>(fsSettings["slam_mode"]))==0) ? SystemMode::VO : SystemMode::VIO;
    fsSettings.release();
  }
  //friend std::ostream &operator<<(std::ostream &os, const Config &cfg);

  const std::string config_file_path_;

  Parameters parameters_;
};