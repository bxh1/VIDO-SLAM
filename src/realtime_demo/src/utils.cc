#include "utils.h"

Config::Config(const std::string &config_file_path)
{
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


