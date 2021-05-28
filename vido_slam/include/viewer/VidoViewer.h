#pragma once
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pangolin/geometry/geometry.h>
#include <pangolin/pangolin.h>
#include <thread>
#include <unistd.h>
#include "System.h"
#include<atomic>
//#include "pangolin_utils/util.h"

namespace VIDO_SLAM
{

struct SceneObject;
class MapViewer {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MapViewer() = delete;
  MapViewer(const std::string &model_path,const std::string &font_path,const int width,const int height);
  ~MapViewer();

  void Run();
  void SetCurrentPose(const cv::Mat &Twc);
  void SetMapPoints(const std::vector<std::vector<cv::Mat>>&mps);
  void SetObjects(const std::vector<SceneObject> &objects);
  void DisplayDynamicImage(cv::Mat img);
  void ForceStop();
  bool GetRunstatus();
  void Reset();
  bool GetPauseState() {
    return is_pause_.load();
  }
private:
  void DrawAxis();
  void DrawGround();
  void DrawTrajectory();
  void DrawObjects();
  void DrawMapPoints();
  std::vector<float>Generate3DBoxVert(Eigen::Vector3f& center);
  pangolin::OpenGlMatrix GetCurrentGLMatrix();
  pangolin::OpenGlMatrix GetCarModelMatrix();
  //void LoadgeometryGpu(pangolin::Geometry &geom, pangolin::RenderNode &root);
 
private:
  std::string model_file_;
  std::vector<SceneObject> objects_;
  std::vector<std::vector<cv::Mat>>map_points_;
  std::mutex mutex_img_;
  Eigen::Matrix4d Twc_;
  int width_;
  int height_;
  unsigned char *img_;
  std::atomic_bool is_pause_;
  bool video_img_changed_;
  std::mutex mutex_run_;
  pangolin::GlFont *glfont_;
  bool running_;
  std::vector<Eigen::Matrix4d> trajectorys_;
  std::mutex mutex_pose_;
  std::mutex mutex_point_;
  std::vector<cv::Mat> particl_;
  std::mutex mutexparticl_;
};
}