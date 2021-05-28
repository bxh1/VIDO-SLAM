#include <System.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<unistd.h>
#include<memory>
#include<opencv2/core/core.hpp>
#include<opencv2/optflow.hpp>
#include "utils.h"
#include "ImuTypes.h"


void LoadIMU(const std::string &strImuPath, std::vector<double> &vTimeImustamps, std::vector<cv::Point3f> &vAcc, std::vector<cv::Point3f> &vGyro)
{
    std::ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeImustamps.reserve(200000);
    vAcc.reserve(200000);
    vGyro.reserve(200000);
    while(!fImu.eof()) {
       std::string s;
       getline(fImu, s);
       if (s[0] == '#')
            continue;
       if(!s.empty())
       {
          std::istringstream line(s);
          std::vector<double>datas;
          std::string field;
          while(std::getline(line, field, ',')) 
          {
            std::stringstream ss;
            ss << field;
            double data;
            ss >> data;
            datas.push_back(data); 
          }
          vTimeImustamps.push_back(datas[0]/1e9);
          vAcc.push_back(cv::Point3f(datas[11],datas[12],datas[13]));
          vGyro.push_back(cv::Point3f(datas[8],datas[9],datas[10]));
       }
    }
    
}

void LoadKaistImg(const std::string &image_dir, std::vector<std::string> &image_names,std::vector<double> &vTimestampsImage) {
  const std::string time_file = image_dir + "/../vTimestampsImage.txt";
  std::ifstream fin(time_file);
  if (!fin.is_open()) {
    std::cout << "vTimestampsImage file open failed from " << time_file << std::endl;
    return;
  }
  std::string line;
  getline(fin, line);
  while (getline(fin, line) && !line.empty()) {
    std::stringstream ss;
    ss << line;
    long double s;
    ss >> s;
    image_names.push_back(std::to_string(s).substr(0,19)+".png");
    double time = s/1e9;
    vTimestampsImage.push_back(time);
  }
}

int main(int argc, char **argv){
   
   if(argc != 2)
   {
         std::cerr << std::endl << "Usage: ./vido_slam path_to_config" << std::endl;
         return 1;
   }
   const std::string config_file = argv[1];
   auto cfg = std::make_shared<Config>(config_file);
   cv::Mat image_trajectory = cv::Mat::zeros(800, 600, CV_8UC3);
   std::vector<std::string> images_names;
   std::vector<double> vTimestampsImage;
   std::map<int,std::vector<VIDO_SLAM::IMU::Point>>vImuMeas;
   std::vector<int> nImages;
   std::vector<int> nImu;
   VIDO_SLAM::System vido_system_vo;
   VIDO_SLAM::System vido_system_vio;
   LoadKaistImg(cfg->parameters_.img_path_, images_names, vTimestampsImage);
   if(cfg->parameters_.slam_mode_==VIO){
      std::vector<cv::Point3f> vAcc, vGyro;
      std::vector<double> vTimestampsImu;
      std::cout<<"load imu data, waiting........."<<std::endl;
      LoadIMU(cfg->parameters_.imu_path_,vTimestampsImu,vAcc,vGyro);
      double lastImageTime;
      for (int idx = 0; idx < vTimestampsImage.size(); ++idx){
          std::vector<VIDO_SLAM::IMU::Point> vImuMea;
          if(idx==0) {
             lastImageTime=vTimestampsImage[0];
          }else{
             std::cout.precision(17);
             for(int iidx=0;iidx<vTimestampsImu.size();++iidx){
                if(lastImageTime<=vTimestampsImu[iidx] && vTimestampsImu[iidx]<=vTimestampsImage[idx]){
                   vImuMea.push_back(VIDO_SLAM::IMU::Point(vAcc[iidx].x,vAcc[iidx].y,vAcc[iidx].z,vGyro[iidx].x,vGyro[iidx].y,vGyro[iidx].z,vTimestampsImu[iidx]));
                }
             }
             vImuMeas.insert(std::make_pair(idx,vImuMea));
             lastImageTime=vTimestampsImage[idx];
          }
      }
      std::cout<<"load imu data done."<<std::endl;
   }
   if(cfg->parameters_.slam_mode_==VO)
      vido_system_vo.Init(config_file,VIDO_SLAM::System::RGBD);
   else
      vido_system_vio.Init(config_file,VIDO_SLAM::System::IMU_RGBD);
   for (int idx = cfg->parameters_.start_image_index_; idx < vTimestampsImage.size(); ++idx) {
      std::cout<<"\nprocessing image idx --> "<<idx<<std::endl;
      cv::Mat raw_img = cv::imread(cfg->parameters_.img_path_ + "/"+images_names[idx],
                             cv::IMREAD_UNCHANGED);
      cv::Mat bgr_img;
      cv::cvtColor(raw_img,bgr_img,cv::COLOR_BayerRG2BGR);
      cv::Mat flow_img = cv::optflow::readOpticalFlow(cfg->parameters_.img_path_ + "/../flow_image/"+images_names[idx].substr(0,19)+".flo");
      cv::Mat depth_img = cv::imread(cfg->parameters_.img_path_ + "/../depth_image/"+images_names[idx].substr(0,19)+".png",CV_LOAD_IMAGE_ANYDEPTH);
      depth_img.convertTo(depth_img,CV_32F);
      cv::Mat mask_img = cv::imread(cfg->parameters_.img_path_ + "/../mask_image/"+images_names[idx].substr(0,19)+".png",CV_LOAD_IMAGE_UNCHANGED);
      mask_img.convertTo(mask_img,CV_32SC1);
      
      std::vector<std::vector<float> > object_pose_gt;
      cv::Mat ground_truth=cv::Mat::eye(4,4,CV_32F);
      if(cfg->parameters_.slam_mode_==VO){
         vido_system_vo.TrackRGBD(bgr_img,depth_img, flow_img, mask_img, ground_truth, object_pose_gt,
                vTimestampsImage[idx], image_trajectory,10000);
      }
      else{
         int iidx=0;
         std::vector<VIDO_SLAM::IMU::Point> vImuM = vImuMeas[idx];
         
         vido_system_vio.TrackRGBD(bgr_img,depth_img, flow_img, mask_img, vImuM, ground_truth,
                object_pose_gt, vTimestampsImage[idx],image_trajectory,10000);
      }
   }
    return 0;
}