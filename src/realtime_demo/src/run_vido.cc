#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <iostream>
#include <stdio.h>
#include <boost/bind.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <flow_net/FlowNet.h>
#include <cv_bridge/cv_bridge.h>
#include <mask_rcnn/MaskRcnn.h>
#include <mono_depth2/MonoDepth.h>
#include <string>
#include <mutex>
#include <thread>
#include "System.h"
#include <tf2_ros/static_transform_broadcaster.h>
#include <sstream>
#include <opencv2/optflow.hpp>

struct VidoSlamInput {
    cv::Mat raw, flow, depth, mask;
    std::vector<std::vector<float> > object_pose_gt;
    cv::Mat ground_truth;
    double image_time; 

    VidoSlamInput(cv::Mat& _raw, cv::Mat& _flow, cv::Mat& _depth, cv::Mat& _mask, double& _image_time) : 
        raw(_raw),
        flow(_flow),
        image_time(_image_time)

    {
        ground_truth = cv::Mat::eye(4,4,CV_32F);
        _depth.convertTo(depth, CV_32F);
        _mask.convertTo(mask, CV_32SC1);
    }

};

bool is_first_=true;
cv::Mat previous_image_;
cv::Mat current_image_;
cv::Mat scene_flow_image_;
cv::Mat mask_image_;
cv::Mat depth_image_;
ros::ServiceClient flow_net_client_;
ros::ServiceClient mask_rcnn_client_;
ros::ServiceClient mono_depth_client_;
std::queue<std::shared_ptr<VidoSlamInput>> vido_input_;
std::mutex input_mutex_;
std::unique_ptr<VIDO_SLAM::System> vido_system_;
cv::Mat image_trajectory = cv::Mat::zeros(800, 600, CV_8UC3);
std::vector<std::string> mask_rcnn_labels;
std::vector<int> mask_rcnn_label_indexs;

bool CallDepthNet(cv::Mat& current_image, cv::Mat& depth_image,ros::NodeHandle& nh_){
    sensor_msgs::ImagePtr current_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",current_image).toImageMsg();
    mono_depth2::MonoDepth srv;
    srv.request.input_image = *current_msg;
    if(mono_depth_client_.call(srv)){
         if(srv.response.success){
             cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(srv.response.output_image, sensor_msgs::image_encodings::MONO16);
             depth_image = cv_ptr->image;
             return true;
          }
          else{
             ROS_ERROR_STREAM("monodepth service return failed");
             return false;
          }
    }
    else{
          ROS_ERROR_STREAM("monodepth service call failed");
          return false;
    }
}


bool CallMaskNet(cv::Mat& current_image, cv::Mat& mask_image,ros::NodeHandle& nh_){
    mask_rcnn_labels.clear();
    mask_rcnn_label_indexs.clear();
    sensor_msgs::ImagePtr current_msg = cv_bridge::CvImage(std_msgs::Header(),"rgb8",current_image).toImageMsg();
    mask_rcnn::MaskRcnn srv;
    srv.request.input_image = *current_msg;
    if(mask_rcnn_client_.call(srv)){
         if(srv.response.success){
             cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(srv.response.output_mask, sensor_msgs::image_encodings::MONO8);
             mask_image = cv_ptr->image;
             for (std::vector<std::string>::iterator it = srv.response.labels.begin(); it != srv.response.labels.end(); ++it) {
                mask_rcnn_labels.push_back(*it);
             }
             for (std::vector<int>::iterator it = srv.response.label_indexs.begin(); it != srv.response.label_indexs.end(); ++it) {
                mask_rcnn_label_indexs.push_back(*it);
             }
             return true;
          }
          else{
             ROS_ERROR_STREAM("MaskRcnn service return failed");
             return false;
          }
    }
    else{
          ROS_ERROR_STREAM("MaskRcnn service call failed");
          return false;
    }
}

bool CallFlowNet(cv::Mat& current_image,cv::Mat& previous_image, cv::Mat& flow_image,ros::NodeHandle& nh_){
    sensor_msgs::ImagePtr current_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",current_image).toImageMsg();
    sensor_msgs::ImagePtr previous_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", previous_image).toImageMsg();
    flow_net::FlowNet srv;
    srv.request.previous_image = *previous_msg;
    srv.request.current_image = *current_msg; 
    if(flow_net_client_.call(srv)){
         if(srv.response.success){
             cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(srv.response.output_image, sensor_msgs::image_encodings::TYPE_32FC2);
             scene_flow_image_ = cv_ptr->image;
             return true;
          }
          else{
             ROS_ERROR_STREAM("Flownet service return failed");
             return false;
          }
    }
    else{
          ROS_ERROR_STREAM("Flownet service call failed");
          return false;
    }
}

void RunNet(cv::Mat& image,double& img_time,std::string& img_name, ros::NodeHandle& nh)
{
   if(is_first_){
       previous_image_ = image;
       is_first_=false;
       return;
   }
   else{
       current_image_ = image;
       
       //flow net
       flow_net_client_ = nh.serviceClient<flow_net::FlowNet>("FlowNetService");
       ros::service::waitForService("FlowNetService");
       if(!CallFlowNet(current_image_,previous_image_,scene_flow_image_,nh))
           ROS_WARN_STREAM("Could not return flow images");
       
       //mask rcnn
       mask_rcnn_client_ = nh.serviceClient<mask_rcnn::MaskRcnn>("MaskRcnnService");
       ros::service::waitForService("MaskRcnnService");
       if(!CallMaskNet(current_image_,mask_image_,nh))
           ROS_WARN_STREAM("Could not return mask rcnn images");
       
       //monodepth2
       mono_depth_client_ = nh.serviceClient<mono_depth2::MonoDepth>("MonoDepthService");
       ros::service::waitForService("MonoDepthService");
       if(!CallDepthNet(current_image_,depth_image_,nh))
           ROS_WARN_STREAM("Could not return monodepth2 images");
       
       //std::string flow_path="/dataset/kaist/urban39/urban39-pankyo/image/flow_image/";
      // cv::optflow::writeOpticalFlow(flow_path+img_name.substr(0,19)+".flo", scene_flow_image_);
       //cv::imwrite(flow_path+"../depth_image/"+img_name.substr(0,19)+".png",depth_image_);
       //cv::imwrite(flow_path+"../mask_image/"+img_name.substr(0,19)+".png",mask_image_);
       auto input = std::make_shared<VidoSlamInput>(image,scene_flow_image_,depth_image_, mask_image_, img_time);
       input_mutex_.lock();
       vido_input_.push(input);
       input_mutex_.unlock();

       previous_image_ = current_image_;
       
   }
}


void LoadKaistImg(const std::string &image_dir, std::vector<std::string> &image_names,std::vector<double> &times) {
  const std::string time_file = image_dir + "/../times.txt";
  std::ifstream fin(time_file);
  if (!fin.is_open()) {
    std::cout << "times file open failed from " << time_file << std::endl;
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
    times.push_back(time);
  }
}

void LoadKittiImg(const std::string &image_dir,std::vector<std::string> &image_names,std::vector<double> &times) {
  const std::string time_file = image_dir + "/../times.txt";
  std::ifstream fin(time_file);
  if (!fin.is_open()) {
    std::cout << "times file open failed from " << time_file << std::endl;
    return;
  }
  std::string line;
  getline(fin, line);
  while (getline(fin, line) && !line.empty()) {
    std::stringstream ss;
    ss << line;
    double s;
    ss >> s;
    times.push_back(s);
  }
  for(size_t i=0;i<times.size();i++){
      std::stringstream name;
      name<<std::setw(10)<<std::setfill('0')<<std::to_string(i)<<".jpg";
      image_names.push_back(name.str());
  }
}



void RunVidoSlam(ros::NodeHandle& n)
{
 // while(true){
   if (!vido_input_.empty()) {
        std::shared_ptr<VidoSlamInput> input;
        input_mutex_.lock();
        input = vido_input_.front();
        vido_input_.pop();
        input_mutex_.unlock();

        cv::Mat pose =  vido_system_->TrackRGBD(input->raw,input->depth,
                input->flow,
                input->mask,
                input->ground_truth,
                input->object_pose_gt,
                input->image_time,
                image_trajectory,20);

   }
  //}
}


int main(int argc, char **argv)
{
   ros::init(argc, argv, "demo");
   ros::NodeHandle n;

   tf2_ros::StaticTransformBroadcaster static_broadcaster;

    
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.header.stamp = ros::Time::now();
    transform_stamped.header.frame_id = "map";
    transform_stamped.child_frame_id = "odom";
    transform_stamped.transform.translation.x = 0;
    transform_stamped.transform.translation.y = 0;
    transform_stamped.transform.translation.z = 0;
    
    //must provide quaternion!
    transform_stamped.transform.rotation.x = 0;
    transform_stamped.transform.rotation.y = 0;
    transform_stamped.transform.rotation.z = 0;
    transform_stamped.transform.rotation.w = 1;

    static_broadcaster.sendTransform(transform_stamped);

   std::string config_file;
   if(n.getParam("config_file", config_file)){
        ROS_INFO_STREAM("Loaded " << "config_file" << ": " << config_file);
   }
   else{
        ROS_ERROR_STREAM("Failed to load " <<  "config_file" );
        n.shutdown();
   };

   vido_system_ = std::make_unique<VIDO_SLAM::System>(config_file,VIDO_SLAM::System::RGBD);
  // std::thread vido_thread(RunVidoSlam,n);

   auto cfg = std::make_shared<Config>(config_file);
   std::vector<std::string> images_names;
   std::vector<double> times;
   LoadKaistImg(cfg->parameters_.img_path_,images_names,times);

   for (int idx = cfg->parameters_.start_image_index_; idx < times.size(); ++idx) {
      std::cout<<"\n ............processing image idx -----> "<<idx<<"...............\n"<<std::endl;
      cv::Mat raw_img = cv::imread(cfg->parameters_.img_path_ + "/"+images_names[idx],
                             cv::IMREAD_UNCHANGED);
      cv::Mat bgr_img;
      cv::cvtColor(raw_img,bgr_img,cv::COLOR_BayerRG2BGR);
      cv::resize(bgr_img,bgr_img,cv::Size(640,192));
      RunNet(bgr_img,times[idx],images_names[idx],n);
      RunVidoSlam(n);
   }
   ros::spin();
   
    
}