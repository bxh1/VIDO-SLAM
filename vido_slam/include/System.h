/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/VDO_SLAM>
*
**/


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include <memory>
#include<opencv2/core/core.hpp>
#include "Tracking.h"
#include "Map.h"
#include "ImuTypes.h"
namespace VIDO_SLAM
{

using namespace std;

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

struct SceneObject {
    SceneObject(){}
    cv::Point3f pose;
    cv::Point2f velocity;
    double yaw=0;
    int label_index; 
    std::string label;
    int tracking_id;
    SceneObject(const SceneObject& scene_object) :
      pose(scene_object.pose),
      velocity(scene_object.velocity),
      label_index(scene_object.label_index),
      label(scene_object.label),
      tracking_id(scene_object.tracking_id) {}
 };


class Map;
class Tracking;

class System
{
public:

    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_RGBD=3
    };

public:
    System(){};
    // Initialize the SLAM system.
    void Init(const string &strSettingsFile, const eSensor sensor);


    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose.
    cv::Mat TrackRGBD(const cv::Mat &im, cv::Mat &depthmap, const cv::Mat &flowmap, const cv::Mat &masksem,
                      const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt, const double &timestamp,
                      cv::Mat &imTraj, const int &nImage);
    // Input imu
    cv::Mat TrackRGBD(const cv::Mat &im, cv::Mat &depthmap, const cv::Mat &flowmap, const cv::Mat &masksem,const vector<IMU::Point>& vImuMeas,
                      const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt, const double &timestamp,
                      cv::Mat &imTraj, const int &nImage);
    void SaveResultsIJRR2020(const string &filename);

private:

    // Input sensor
    eSensor mSensor;

    // Map structure.
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    Tracking* mpTracker;

};

}// namespace VIDO_SLAM

#endif // SYSTEM_H
