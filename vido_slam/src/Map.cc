/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/VDO_SLAM>
*
**/

#include <iostream>

#include "Map.h"

namespace VIDO_SLAM
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{}

void Map::reset() {
        std::cout << "Resetting map" << std::endl;
        vpFeatSta.clear();
        vfDepSta.clear();
        vp3DPointSta.clear();
        vnAssoSta.clear();
        TrackletSta.clear();
        vpFeatDyn.clear();
        vfDepDyn.clear();
        vp3DPointDyn.clear();
        vnAssoDyn.clear();
        vnFeatLabel.clear();
        TrackletDyn.clear();
        nObjID.clear();
        vmCameraPose.clear();
        vmCameraPose_RF.clear();
        vmCameraPose_GT.clear();
        vmRigidCentre.clear();
        vmRigidMotion.clear();
        vmObjPosePre.clear();
        vmRigidMotion_RF.clear();
        vmRigidMotion_GT.clear();
        vfAllSpeed_GT.clear();
        vnRMLabel.clear();
        vnSMLabel.clear();
        vnSMLabelGT.clear();
        vbObjStat.clear();
        vnObjTraTime.clear();
        nObjTraCount.clear();
        nObjTraCountGT.clear();
        nObjTraSemLab.clear();
        fLBA_time.clear();
        vfAll_time.clear();
        vpFrames.clear();
        mnMaxKFid = 0;
        mnBigChangeIdx = 0;
    }
    
void Map::ApplyScaledRotation(const cv::Mat &R, const float s, const bool bScaledVel, const cv::Mat t)
{
    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    cv::Mat Txw = cv::Mat::eye(4,4,CV_32F);
    R.copyTo(Txw.rowRange(0,3).colRange(0,3));//Rgw

    cv::Mat Tyx = cv::Mat::eye(4,4,CV_32F);

    cv::Mat Tyw = Tyx*Txw;
    Tyw.rowRange(0,3).col(3) = Tyw.rowRange(0,3).col(3)+t;
    cv::Mat Ryw = Tyw.rowRange(0,3).colRange(0,3);
    cv::Mat tyw = Tyw.rowRange(0,3).col(3);

    for(vector<Frame*>::iterator sit=vpFrames.begin(); sit!=vpFrames.end(); sit++)
    {
        Frame* pF = *sit;
        cv::Mat Twc = pF->GetPoseInverse();
        Twc.rowRange(0,3).col(3)*=s;
        cv::Mat Tyc = Tyw*Twc;
        cv::Mat Tcy = cv::Mat::eye(4,4,CV_32F);
        Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
        Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
        pF->SetPose(Tcy);
        cv::Mat Vw = pF->GetVelocity();
        if(!bScaledVel)
            pF->SetVelocity(Ryw*Vw);
        else
            pF->SetVelocity(Ryw*Vw*s);
        for(int i=0;i<pF->mvStat3DPointTmp.size();i++){
            cv::Mat point3d = pF->mvStat3DPointTmp[i];
            pF->mvStat3DPointTmp[i] = s*Ryw*point3d+tyw;
        }
        for(int j=0;j<pF->mvObj3DPoint.size();j++){
            cv::Mat point3d = pF->mvObj3DPoint[j];
            pF->mvObj3DPoint[j] = s*Ryw*point3d+tyw;
        }
    }

    for(int i=0;i<vp3DPointSta.size();i++){
        for(int j=0;j<vp3DPointSta[i].size();j++){
            cv::Mat point3d = vp3DPointSta[i][j];
            vp3DPointSta[i][j] = s*Ryw*point3d+tyw;
        }
    }
    for(int k=0;k<vp3DPointDyn.size();k++){
        for(int m=0;m<vp3DPointDyn[k].size();m++){
            cv::Mat point3d = vp3DPointDyn[k][m];
            vp3DPointDyn[k][m] = s*Ryw*point3d+tyw;
        }
    }
    for(int idx=0;idx<vmCameraPose.size();idx++){
        cv::Mat pose = vmCameraPose[idx];
        pose.rowRange(0,3).col(3)*=s;
        vmCameraPose[idx] = Tyw*pose;
    }
    for(int kk=0;kk<vmRigidMotion.size();kk++){
        for(int mm=0;mm<vmRigidMotion[kk].size();mm++){
            cv::Mat pose = vmRigidMotion[kk][mm];
            pose.rowRange(0,3).col(3)*=s;
            vmRigidMotion[kk][mm] = Tyw*pose;
        }
    }
    
}

} //namespace VIDO_SLAM
