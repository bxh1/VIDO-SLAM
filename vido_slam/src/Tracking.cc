
#include <Eigen/Core>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <cvplot/cvplot.h>

#include"Converter.h"
#include"Map.h"
#include"Optimizer.h"
#include"Tracking.h"
#include<iostream>
#include<string>
#include<stdio.h>
#include<math.h>
#include<time.h>

#include<mutex>
#include<unistd.h>

#include <memory>
#include <numeric>
#include <algorithm>
#include <map>
#include <random>

using namespace std;

bool SortPairInt(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second > b.second);
}

namespace VIDO_SLAM
{

Tracking::Tracking(System *pSys, Map *pMap, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mpSystem(pSys), mpMap(pMap), mScale(1.0),mbImuInitialized(false),mbIMU_BA1(false),
    mbIMU_BA2(false),mpLastFrame(NULL)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    int image_height = fSettings["Camera.height"];
    int image_width = fSettings["Camera.width"];
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;
     // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;
    cout << endl << "Camera Parameters: " << endl << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- fps: " << fps << endl;

    std::string model_path = fSettings["CarModel"];
    std::string font_path = fSettings["FontPath"];
    mapviewer = new MapViewer(model_path,font_path,image_width,image_height);
    std::thread mptviewer = std::thread(&MapViewer::Run, mapviewer);
    mptviewer.detach();
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    
    bool b_parse_imu = true;
    if(sensor==System::IMU_RGBD)
    {
        b_parse_imu = ParseIMUParamFile(fSettings);
        if(!b_parse_imu)
        {
            std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
        }

        mnFramesToResetIMU = mMaxFrames;
    }

    cout << endl << "System Parameters: " << endl << endl;
  
    int DataCode = fSettings["ChooseData"];
    switch (DataCode)
    {
        case 1:
            mTestData = OMD;
            cout << "- tested dataset: OMD " << endl;
            break;
        case 2:
            mTestData = KITTI;
            cout << "- tested dataset: KITTI " << endl;
            break;
        case 3:
            mTestData = KAIST;
            cout << "- tested dataset: KAIST " << endl;
            break;
    }

    if(sensor==System::IMU_RGBD || sensor==System::RGBD)
    {
        mThDepth = (float)fSettings["ThDepthBG"];
        mThDepthObj = (float)fSettings["ThDepthOBJ"];
        cout << "- depth threshold (background/object): " << mThDepth << "/" << mThDepthObj << endl;
    }

    if(sensor==System::IMU_RGBD || sensor==System::RGBD )
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        cout << "- depth map factor: " << mDepthMapFactor << endl;
    }

    nMaxTrackPointBG = fSettings["MaxTrackPointBG"];
    nMaxTrackPointOBJ = fSettings["MaxTrackPointOBJ"];
    cout << "- max tracking points: " << "(1) background: " << nMaxTrackPointBG << " (2) object: " << nMaxTrackPointOBJ << endl;

    fSFMgThres = fSettings["SFMgThres"];
    fSFDsThres = fSettings["SFDsThres"];
    cout << "- scene flow paras: " << "(1) magnitude: " << fSFMgThres << " (2) percentage: " << fSFDsThres << endl;

    nWINDOW_SIZE = fSettings["WINDOW_SIZE"];
    nOVERLAP_SIZE = fSettings["OVERLAP_SIZE"];
    cout << "- local batch paras: " << "(1) window: " << nWINDOW_SIZE << " (2) overlap: " << nOVERLAP_SIZE << endl;

    nUseSampleFea = fSettings["UseSampleFeature"];
    if (nUseSampleFea==1)
        cout << "- used sampled feature for background scene..." << endl;
    else
        cout << "- used detected feature for background scene..." << endl;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat Tbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        Tbc = node.mat();
        if(Tbc.rows != 4 || Tbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }

    cout << endl;

    cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

    float freq, Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        freq = node.operator int();
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(freq);
    cout << endl;
    cout << "IMU frequency: " << freq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng,Na,Ngw,Naw);

    mpImuPreintegratedFrompLastFrame = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, cv::Mat &imD, const cv::Mat &imFlow,
                                const cv::Mat &maskSEM, const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt,
                                const double &timestamp, cv::Mat &imTraj, const int &nImage)
{
    // initialize some paras
    StopFrame = nImage-1;
    cv::RNG rng((unsigned)time(NULL));

    // Initialize Global ID
    if (mState==NO_IMAGES_YET) {
        f_id = 0;
    }

    mImGray = imRGB;

    // preprocess depth 
    for (int i = 0; i < imD.rows; i++)
    {
        for (int j = 0; j < imD.cols; j++)
        {
            if (imD.at<float>(i,j)<0)
                imD.at<float>(i,j)=0;
            else
            {
                if (mTestData==OMD)
                    imD.at<float>(i,j) = imD.at<float>(i,j)/mDepthMapFactor;
                else if (mTestData==KITTI)
                {
                    // --- for stereo depth map --
                    imD.at<float>(i,j) = mbf/(imD.at<float>(i,j)/mDepthMapFactor);
                    // --- for monocular depth map ---
                    // imD.at<float>(i,j) = imD.at<float>(i,j)/500.0;
                }
                else if(mTestData == KAIST)
                {
                    imD.at<float>(i,j) = mScale * mbf/(imD.at<float>(i,j)/mDepthMapFactor);
                }
            }
        }
    }

    cv::Mat imDepth = imD;

    // Transform color image to grey image
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Save map in the tracking head 
    mDepthMap = imD;
    mFlowMap = imFlow;
    mSegMap = maskSEM;

    // Initialize timing vector (Output)
    all_timing.resize(5,0);

    if (mState!=NO_IMAGES_YET)
    {
        clock_t s_0, e_0;
        double mask_upd_time;
        s_0 = clock();
        // ****** Update Mask information *******
        UpdateMask();
        e_0 = clock();
        mask_upd_time = (double)(e_0-s_0)/CLOCKS_PER_SEC*1000;
        all_timing[0] = mask_upd_time;
        // cout << "mask updating time: " << mask_upd_time << endl;
    }
    if(mSensor==System::RGBD)
       mpCurrentFrame = new Frame(mImGray,imDepth,imFlow,maskSEM,timestamp,mpORBextractorLeft,mK,mDistCoef,mbf,mThDepth,mThDepthObj,nUseSampleFea);
    else
       mpCurrentFrame = new Frame(isImuInitialized(),mpLastFrame, mImGray,imDepth,imFlow,maskSEM,*mpImuCalib,timestamp,mpORBextractorLeft,mK,mDistCoef,mbf,mThDepth,mThDepthObj,nUseSampleFea);
    // ---------------------------------------------------------------------------------------
    // +++++++++++++++++++++++++ For sampled features ++++++++++++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------
    if(mState!=NO_IMAGES_YET)
    {

        mpCurrentFrame->mvStatKeys = mpLastFrame->mvCorres;//当前帧对应上一帧的静态匹配点
        mpCurrentFrame->N_s = mpCurrentFrame->mvStatKeys.size();
        // assign the depth value to each keypoint
        mpCurrentFrame->mvStatDepth = std::vector<float>(mpCurrentFrame->N_s,-1);
        for(int i=0; i<mpCurrentFrame->N_s; i++)
        {
            const cv::KeyPoint &kp = mpCurrentFrame->mvStatKeys[i];

            const int v = kp.pt.y;
            const int u = kp.pt.x;

            if (u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v>0)
            {
                float d = imDepth.at<float>(v,u); // be careful with the order  !!!

                if(d>0)
                    mpCurrentFrame->mvStatDepth[i] = d;
            }

        }

        // *********** Save object keypoints and depths ************

        // *** first assign current keypoints and depth to last frame
        // *** then assign last correspondences to current frame
        
        mvTmpObjKeys = mpCurrentFrame->mvObjKeys;
        mvTmpObjDepth = mpCurrentFrame->mvObjDepth;
        mvTmpSemObjLabel = mpCurrentFrame->vSemObjLabel;
        mvTmpObjFlowNext = mpCurrentFrame->mvObjFlowNext;
        mvTmpObjCorres = mpCurrentFrame->mvObjCorres;

        mpCurrentFrame->mvObjKeys = mpLastFrame->mvObjCorres;
        mpCurrentFrame->mvObjDepth.resize(mpCurrentFrame->mvObjKeys.size(),-1);
        mpCurrentFrame->vSemObjLabel.resize(mpCurrentFrame->mvObjKeys.size(),-1);
        for (int i = 0; i < mpCurrentFrame->mvObjKeys.size(); ++i)
        {
            const int u = mpCurrentFrame->mvObjKeys[i].pt.x;
            const int v = mpCurrentFrame->mvObjKeys[i].pt.y;
            if (u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v>0 && imDepth.at<float>(v,u)<mThDepthObj && imDepth.at<float>(v,u)>0)
            {
                mpCurrentFrame->mvObjDepth[i] = imDepth.at<float>(v,u);
                mpCurrentFrame->vSemObjLabel[i] = maskSEM.at<int>(v,u);
            }
            else
            {
                mpCurrentFrame->mvObjDepth[i] = 0.1;
                mpCurrentFrame->vSemObjLabel[i] = 0;
            }
        }

        // **********************************************************
        // show image
        //cv::Mat img_show;
        //cv::drawKeypoints(mImGray, mpCurrentFrame->mvObjKeys, img_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        //cv::imshow("Dense Feature Distribution 2", img_show);
        //cv::waitKey(1);
        //cout << "Update Current Frame, Done!" << endl;
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    // // Assign pose ground truth
    if (mState==NO_IMAGES_YET)
    {
        mpCurrentFrame->mTcw_gt = Converter::toInvMatrix(mTcw_gt);
        mOriginInv = mTcw_gt;
    }
    else
    {
       // mpCurrentFrame->mTcw_gt = Converter::toInvMatrix(mTcw_gt)*mOriginInv;
    }


    // Save temperal matches for visualization
    if (mState!=NO_IMAGES_YET)
        TemperalMatch = vector<int>(mpCurrentFrame->N_s,-1);
    // Initialize object label
    mpCurrentFrame->vObjLabel.resize(mpCurrentFrame->mvObjKeys.size(),-2);

    // *** main ***
    Track();
    // ************

    // Update Global ID
    f_id = f_id + 1;

    // ---------------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++++++ Display Information ++++++++++++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------------

    // // // ************** display label on the image ***************  // //
    if(timestamp!=0 && bFrame2Frame == true)
    {
        std::vector<cv::KeyPoint> KeyPoints_tmp(1);
        // background features
        // for (int i = 0; i < mpCurrentFrame->mvStatKeys.size(); i=i+1)
        // {
        //     KeyPoints_tmp[0] = mpCurrentFrame->mvStatKeys[i];
        //     if(maskSEM.at<int>(KeyPoints_tmp[0].pt.y,KeyPoints_tmp[0].pt.x)!=0)
        //         continue;
        //     cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
        // }
        for (int i = 0; i < TemperalMatch_subset.size(); i=i+1)
        {
            if (TemperalMatch_subset[i]>=mpCurrentFrame->mvStatKeys.size())
                continue;
            KeyPoints_tmp[0] = mpCurrentFrame->mvStatKeys[TemperalMatch_subset[i]];
            if (KeyPoints_tmp[0].pt.x>=(mImGray.cols-1) || KeyPoints_tmp[0].pt.x<=0 || KeyPoints_tmp[0].pt.y>=(mImGray.rows-1) || KeyPoints_tmp[0].pt.y<=0)
                continue;
            if(maskSEM.at<int>(KeyPoints_tmp[0].pt.y,KeyPoints_tmp[0].pt.x)!=0)
                continue;
            cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(200,255,0), 1); 
        }
        // static and dynamic objects
        for (int i = 0; i < mpCurrentFrame->vObjLabel.size(); ++i)
        {
            if(mpCurrentFrame->vObjLabel[i]==-1 || mpCurrentFrame->vObjLabel[i]==-2)
                continue;
            int l = mpCurrentFrame->vObjLabel[i];
            if (l>25)
                l = l/2;
            // int l = mpCurrentFrame->vSemObjLabel[i];
            // cout << "label: " << l << endl;
            KeyPoints_tmp[0] = mpCurrentFrame->mvObjKeys[i];
            if (KeyPoints_tmp[0].pt.x>=(mImGray.cols-1) || KeyPoints_tmp[0].pt.x<=0 || KeyPoints_tmp[0].pt.y>=(mImGray.rows-1) || KeyPoints_tmp[0].pt.y<=0)
                continue;
            switch (l)
            {
                case 0:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
                    break;
                case 1:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1); // 255, 165, 0
                    break;
                case 2:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,0), 1);
                    break;
                case 3:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 255, 0), 1); // 255,255,0
                    break;
                case 4:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,0,0), 1); // 255,192,203
                    break;
                case 5:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,255), 1);
                    break;
                case 6:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1);
                    break;
                case 7:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,255), 1);
                    break;
                case 8:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,228,196), 1);
                    break;
                case 9:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
                    break;
                case 10:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(165,42,42), 1);
                    break;
                case 11:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(35, 142, 107), 1);
                    break;
                case 12:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(45, 82, 160), 1);
                    break;
                case 13:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
                    break;
                case 14:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 165, 0), 1);
                    break;
                case 15:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,0), 1);
                    break;
                case 16:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,0), 1);
                    break;
                case 17:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,192,203), 1);
                    break;
                case 18:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,255), 1);
                    break;
                case 19:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1);
                    break;
                case 20:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,255), 1);
                    break;
                case 21:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,228,196), 1);
                    break;
                case 22:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
                    break;
                case 23:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(165,42,42), 1);
                    break;
                case 24:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(35, 142, 107), 1);
                    break;
                case 25:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(45, 82, 160), 1);
                    break;
                case 41:
                    cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(60, 20, 220), 1);
                    break;
            }
        }
        //cv::imshow("Static Background and Object Points", imRGB);
        mapviewer->DisplayDynamicImage(imRGB);
        // cv::imwrite("feat.png",imRGB);
        if (f_id<4)
            cv::waitKey(1);
        else
            cv::waitKey(1);

    }
    // ************** show bounding box with speed ***************
    // if(timestamp!=0 && bFrame2Frame == true && mTestData==KITTI)
    // {
    //     cout << "Showing bb with speed" << endl;
    //     cv::Mat mImBGR(mImGray.size(), CV_8UC3);
    //     cvtColor(mImGray, mImBGR, CV_GRAY2RGB);
    //     cout << "v obj box size " << mpCurrentFrame->vObjBoxID.size() << endl;
    //     for (int i = 0; i < mpCurrentFrame->vObjBoxID.size(); ++i)
    //     {
    //         if (mpCurrentFrame->vSpeed[i].x==0)
    //             continue;
    //         cout << "ID: " << mpCurrentFrame->vObjBoxID[i] << endl;
    //         cv::Point pt1(vObjPose_gt[mpCurrentFrame->vObjBoxID[i]][2], vObjPose_gt[mpCurrentFrame->vObjBoxID[i]][3]);
    //         cv::Point pt2(vObjPose_gt[mpCurrentFrame->vObjBoxID[i]][4], vObjPose_gt[mpCurrentFrame->vObjBoxID[i]][5]);
    //         // cout << pt1.x << " " << pt1.y << " " << pt2.x << " " << pt2.y << endl;
    //         cv::rectangle(mImBGR, pt1, pt2, cv::Scalar(0, 255, 0),2);
    //         // string sp_gt = std::to_string(mpCurrentFrame->vSpeed[i].y);
    //         string sp_est = std::to_string(mpCurrentFrame->vSpeed[i].x/36);
    //         // sp_gt.resize(5);
    //         sp_est.resize(5);
    //         // string output_gt = "GT:" + sp_gt + "km/h";
    //         string output_est = sp_est + "km/h";
    //         cv::putText(mImBGR, output_est, cv::Point(pt1.x, pt1.y-10), cv::FONT_HERSHEY_DUPLEX, 0.9, CV_RGB(0,255,0), 2); // CV_RGB(255,140,0)
    //         // cv::putText(mImBGR, output_gt, cv::Point(pt1.x, pt1.y-32), cv::FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255, 0, 0), 2);
    //     }
    //     cv::imshow("Object Speed", mImBGR);
    //     cv::waitKey(1);
    // }

    // // ************** show trajectory results ***************
    if (!mpCurrentFrame->mTcw.empty())
    {
        int sta_x = 300, sta_y = 100, radi = 2, thic = 5;  // (160/120/2/5)
        float scale = 6; // 6
        cv::Mat CamPos = Converter::toInvMatrix(mpCurrentFrame->mTcw);
        int x = int(CamPos.at<float>(0,3)*scale) + sta_x;
        int y = int(CamPos.at<float>(2,3)*scale) + sta_y;
        // cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,0,0), thic);
        cv::rectangle(imTraj, cv::Point(x, y), cv::Point(x+10, y+10), cv::Scalar(0,0,255),1);
        cv::rectangle(imTraj, cv::Point(10, 30), cv::Point(550, 60), CV_RGB(0,0,0), CV_FILLED);
        cv::putText(imTraj, "Camera Trajectory (RED SQUARE)", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);
        char text[100];
        sprintf(text, "x = %02fm y = %02fm z = %02fm", CamPos.at<float>(0,3), CamPos.at<float>(1,3), CamPos.at<float>(2,3));

        mapviewer->SetCurrentPose(CamPos);

        cv::putText(imTraj, text, cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar::all(255), 1);
        cv::putText(imTraj, "Object Trajectories (COLORED CIRCLES)", cv::Point(10, 70), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);
        std::vector<SceneObject> objects;
        for (int i = 0; i < mpCurrentFrame->vObjCentre3D.size(); ++i)
        {
            if (mpCurrentFrame->vObjCentre3D[i].at<float>(0,0)==0 && mpCurrentFrame->vObjCentre3D[i].at<float>(0,2)==0) {
                continue;
            }
            int x = int(mpCurrentFrame->vObjCentre3D[i].at<float>(0,0)*scale) + sta_x;
            int y = int(mpCurrentFrame->vObjCentre3D[i].at<float>(0,2)*scale) + sta_y;

            float world_x = mpCurrentFrame->vObjCentre3D[i].at<float>(0,0);
            float world_y = mpCurrentFrame->vObjCentre3D[i].at<float>(0,1);
            float world_z = mpCurrentFrame->vObjCentre3D[i].at<float>(0,2);

            float vel_x = mpCurrentFrame->vSpeed[i].x/36;
            float vel_y = mpCurrentFrame->vSpeed[i].y/36;
            // int l = mpCurrentFrame->nSemPosition[i];
            int l = mpCurrentFrame->nModLabel[i];

            SceneObject scene_object;
            scene_object.pose = cv::Point3f(world_x, world_y, world_z);
            scene_object.velocity = cv::Point2f(vel_x, vel_y);
            scene_object.tracking_id = l;
            scene_object.label_index = mpCurrentFrame->nSemPosition[i];
            objects.push_back(scene_object);
            if(!objects.empty())
            {
                for(SceneObject& object : objects){
                if(object.tracking_id==l){
                   double dx = object.pose.x-scene_object.pose.x;
                   double dz = object.pose.z-scene_object.pose.z;
                   double yaw = atan2(dx,dz);
                   scene_object.yaw=yaw;
                }
              }
            }
            switch (l)
            {
                case 1:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic); // orange
                    break;
                case 2:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0,255,255), thic); // green
                    break;
                case 3:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0, 255, 0), thic); // yellow
                    break;
                case 4:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0,0,255), thic); // pink
                    break;
                case 5:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,255,0), thic); // cyan (yellow green 47,255,173)
                    break;
                case 6:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic); // purple
                    break;
                case 7:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,255,255), thic);  // white
                    break;
                case 8:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(196,228,255), thic); // bisque
                    break;
                case 9:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(180, 105, 255), thic);  // blue
                    break;
                case 10:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(42,42,165), thic);  // brown
                    break;
                case 11:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(35, 142, 107), thic);
                    break;
                case 12:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(45, 82, 160), thic);
                    break;
                case 41:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(60, 20, 220), thic);
                    break;
            }
        }
        mapviewer->SetObjects(objects);
        mapviewer->SetMapPoints(mpMap->vp3DPointSta);
        imshow( "Camera and Object Trajectories", imTraj);
        if (f_id<3)
            cv::waitKey(1);
        else
            cv::waitKey(1);
    }
 
    while(mapviewer->GetPauseState()) {
       std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // if(timestamp!=0 && bFrame2Frame == true && mTestData==OMD)
    // {
    //     PlotMetricError(mpMap->vmCameraPose,mpMap->vmRigidMotion, mpMap->vmObjPosePre,
    //                    mpMap->vmCameraPose_GT,mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
    // }


    // // // ************** display temperal matching ***************
    // if(timestamp!=0 && bFrame2Frame == true)
    // {
    //     std::vector<cv::KeyPoint> PreKeys, CurKeys;
    //     std::vector<cv::DMatch> TemperalMatches;
    //     int count =0;
    //     for(int iL=0; iL<mvKeysCurrentFrame.size(); iL=iL+50)
    //     {
    //         if(maskSEM.at<int>(mvKeysCurrentFrame[iL].pt.y,mvKeysCurrentFrame[iL].pt.x)!=0)
    //             continue;
    //         // if(TemperalMatch[iL]==-1)
    //         //     continue;
    //         // if(checkit[iL]==0)
    //         //     continue;
    //         // if(mpCurrentFrame->vObjLabel[iL]<=0)
    //         //     continue;
    //         // if(cv::norm(mpCurrentFrame->vFlow_3d[iL])<0.15)
    //         //     continue;
    //         PreKeys.push_back(mvKeysLastFrame[TemperalMatch[iL]]);
    //         CurKeys.push_back(mvKeysCurrentFrame[iL]);
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }
    //     // cout << "temperal features numeber: " << count <<  endl;

    //     cv::Mat img_matches;
    //     drawMatches(mImGrayLast, PreKeys, mImGray, CurKeys,
    //                 TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
    //                 vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //     cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/1.0, img_matches.rows/1.0));
    //     cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    //     cv::imshow("temperal matches", img_matches);
    //     cv::waitKey(0);
    // }

    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------

    mImGrayLast = mImGray;
    TemperalMatch.clear();
    mSegMapLast = mSegMap;   
    mFlowMapLast = mFlowMap;
    return mpCurrentFrame->mTcw.clone();
}

void Tracking::PreintegrateIMU()
{
   // cout << "start preintegration" << endl;

    if(!mpCurrentFrame->mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mpCurrentFrame->setIntegrated();
        return;
    }

    mvImuFrompLastFrame.clear();
    mvImuFrompLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mpCurrentFrame->setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mpCurrentFrame->mpPrevFrame->mTimeStamp-0.001l)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mpCurrentFrame->mTimeStamp-0.001l)
                {
                    mvImuFrompLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFrompLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }


    const int n = mvImuFrompLastFrame.size()-1;
    IMU::Preintegrated* pImuPreintegratedFrompLastFrame = new IMU::Preintegrated(mpLastFrame->mImuBias,mpCurrentFrame->mImuCalib);
    for(int i=0; i<n; i++)
    {
        float tstep;
        cv::Point3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFrompLastFrame[i+1].t-mvImuFrompLastFrame[i].t;
             // 获取当前imu到上一帧的时间间隔
            float tini = mvImuFrompLastFrame[i].t-mpCurrentFrame->mpPrevFrame->mTimeStamp;
            // 差值计算上一帧到当前时刻imu的一个平均加速度。imu时间不会正好落在上一帧的时刻，需要做补偿，要求得a0时刻到上一帧这段时间加速度的改变量
            // 有了这个改变量将其加到a0上之后就可以表示上一帧时的加速度了。其中a0 - (a1-a0)*(tini/tab) 为上一帧时刻的加速度，再加上a1 之后除以2就为这段时间的加速度平均值
            // 其中tstep表示a1到上一帧的时间间隔
            acc = (mvImuFrompLastFrame[i].a+mvImuFrompLastFrame[i+1].a-
                    (mvImuFrompLastFrame[i+1].a-mvImuFrompLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFrompLastFrame[i].w+mvImuFrompLastFrame[i+1].w-
                    (mvImuFrompLastFrame[i+1].w-mvImuFrompLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFrompLastFrame[i+1].t-mpCurrentFrame->mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFrompLastFrame[i].a+mvImuFrompLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFrompLastFrame[i].w+mvImuFrompLastFrame[i+1].w)*0.5f;
            tstep = mvImuFrompLastFrame[i+1].t-mvImuFrompLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFrompLastFrame[i+1].t-mvImuFrompLastFrame[i].t;
            float tend = mvImuFrompLastFrame[i+1].t-mpCurrentFrame->mTimeStamp;
            acc = (mvImuFrompLastFrame[i].a+mvImuFrompLastFrame[i+1].a-
                    (mvImuFrompLastFrame[i+1].a-mvImuFrompLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFrompLastFrame[i].w+mvImuFrompLastFrame[i+1].w-
                    (mvImuFrompLastFrame[i+1].w-mvImuFrompLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mpCurrentFrame->mTimeStamp-mvImuFrompLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFrompLastFrame[i].a;
            angVel = mvImuFrompLastFrame[i].w;
            tstep = mpCurrentFrame->mTimeStamp-mpCurrentFrame->mpPrevFrame->mTimeStamp;
        }
        pImuPreintegratedFrompLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }
    mpCurrentFrame->mpImuPreintegrated = pImuPreintegratedFrompLastFrame;
    mpCurrentFrame->setIntegrated();
    Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);

}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, Frame* pCurrentFrame)
{
    mLastBias = b;

    mpLastFrame = pCurrentFrame;

    mpLastFrame->SetNewBias(mLastBias);
    mpCurrentFrame->SetNewBias(mLastBias);

    cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

    cv::Mat twb1; 
    twb1 = mpLastFrame->mpPrevFrame->GetImuPosition();
    cv::Mat Rwb1;
    Rwb1 = mpLastFrame->mpPrevFrame->GetImuRotation();
    cv::Mat Vwb1; 
    Vwb1 = mpLastFrame->mpPrevFrame->GetVelocity();
    float t12;
    t12 = mpLastFrame->mpImuPreintegrated->dT;
    mpLastFrame->SetImuPoseVelocity(Rwb1*mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                    twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                    Vwb1 + Gz*t12 + Rwb1*mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());

    if (mpCurrentFrame->mpImuPreintegrated)
    {
        twb1 = mpCurrentFrame->mpPrevFrame->GetImuPosition();
        Rwb1 = mpCurrentFrame->mpPrevFrame->GetImuRotation();
        Vwb1 = mpCurrentFrame->mpPrevFrame->GetVelocity();
        t12 = mpCurrentFrame->mpImuPreintegrated->dT;
        mpCurrentFrame->SetImuPoseVelocity(Rwb1*mpCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                          twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                          Vwb1 + Gz*t12 + Rwb1*mpCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
    mnFirstImuFrameId = mpCurrentFrame->mnId;
}



bool Tracking::isImuInitialized()
{
   return mbImuInitialized;
}

void Tracking::SetImuInitialized()
{
   mbImuInitialized = true;
}

void Tracking::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    float minTime = 2.0;
    int nMinF = 10;


    if(mpMap->GetFramesInMapSize()<nMinF)
        return;

    vector<Frame*> vpF = mpMap->GetFramesInMap();
    mFirstTs=vpF.front()->mTimeStamp;
    if(mpCurrentFrame->mTimeStamp-mFirstTs<minTime)
        return;


    const int N = vpF.size();
    IMU::Bias b(0,0,0,0,0,0);
    // Compute Frame velocities and mRwg estimation
    if (!isImuInitialized())
    {
        cv::Mat cvRwg;
        cv::Mat dirG = cv::Mat::zeros(3,1,CV_32F);
        for(vector<Frame*>::iterator itF = vpF.begin(); itF!=vpF.end(); itF++)
        {
            if (!(*itF)->mpImuPreintegrated)
                continue;
            if (!(*itF)->mpPrevFrame)
                continue;
            // Rwb（imu坐标转到初始化前世界坐标系下的坐标）*更新偏置后的速度，可以理解为在世界坐标系下的速度矢量
            dirG -= (*itF)->mpPrevFrame->GetImuRotation()*(*itF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();//世界坐标系下速度负值累加
            cv::Mat _vel = ((*itF)->GetImuPosition() - (*itF)->mpPrevFrame->GetImuPosition())/(*itF)->mpImuPreintegrated->dT;
            (*itF)->SetVelocity(_vel);
            (*itF)->mpPrevFrame->SetVelocity(_vel);
        }

        dirG = dirG/cv::norm(dirG);
        cv::Mat gI = (cv::Mat_<float>(3,1) << 0.0f, 0.0f, -1.0f);
        //速度方向与重力方向的角轴
        cv::Mat v = gI.cross(dirG);
        const float nv = cv::norm(v);
        //转角大小
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        //旋转向量
        cv::Mat vzg = v*ang/nv;
        //重力方向到当前速度方向的旋转向量
        cvRwg = IMU::ExpSO3(vzg);
        mRwg = Converter::toMatrix3d(cvRwg);
        mTinit = mpCurrentFrame->mTimeStamp-mFirstTs;
        mbg = Converter::toVector3d(mpCurrentFrame->GetGyroBias());
        mba = Converter::toVector3d(mpCurrentFrame->GetAccBias());
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = Converter::toVector3d(mpCurrentFrame->GetGyroBias());
        mba = Converter::toVector3d(mpCurrentFrame->GetAccBias());
    }
    std::cout<<"mba : "<<mba<<endl;
    std::cout<<"mbg : "<<mbg<<endl;
    std::cout<<"mRwg : "<<mRwg<<endl;
    mScale=1.0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpMap, mRwg, mScale, mbg, mba, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    cout << "scale after inertial-only optimization: " << mScale << endl;
    cout << "bg after inertial-only optimization: " << mbg << endl;
    cout << "ba after inertial-only optimization: " << mba << endl;

    
    if (mScale<1e-1)
    {
        cout << "scale is too small: " <<mScale<< endl;
        return;
    }

    // Before this line we are not changing the map
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.00001))
    {
        mpMap->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
        UpdateFrameIMU(mScale,vpF[0]->GetImuBias(),mpCurrentFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    
    for(int i=0;i<N;i++)
    {
       Frame* pF2 = vpF[i];
       pF2->mbImu = true;
     }

    /*cout << "Before GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/


    if (!isImuInitialized())
    {
        cout << "IMU is initialized" << endl;
        SetImuInitialized();
        t0IMU = mpCurrentFrame->mTimeStamp;
        mpCurrentFrame->mbImu = true;
    }
    mState=Tracking::OK;

    return;
}

void Tracking::ScaleRefinement()
{
    vector<Frame*> vpF = mpMap->GetFramesInMap();
    const int N = vpF.size();
    IMU::Bias b(0,0,0,0,0,0);
    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpMap, mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    cout << "scale refine ,scale after inertial-only optimization: " << mScale << endl;
    cout << "bg after inertial-only optimization: " << mbg << endl;
    cout << "ba after inertial-only optimization: " << mba << endl;


    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        return;
    }

    // Before this line we are not changing the map
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.00001))
    {
        mpMap->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
        UpdateFrameIMU(mScale,mpCurrentFrame->GetImuBias(),mpCurrentFrame);
    }

    return;
}



void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState=mState;


    if(mState==NOT_INITIALIZED)
    {
        bFirstFrame = true;
        bFrame2Frame = false;

        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            Initialization();

        if(mState!=OK)
            return;
    }
    else
    {
        bFrame2Frame = true;

        // // *********** Update TemperalMatch ***********
        for (int i = 0; i < mpCurrentFrame->N_s; ++i){
            TemperalMatch[i] = i;
        }
        // // ********************************************

        if (TemperalMatch.size() < 2) {
            cout << "Temperal Match size is < 2" << endl;
            return;
        }

        if(mSensor==System::IMU_RGBD){
            IMU::Bias b = mpLastFrame->GetImuBias();
            mpCurrentFrame->SetNewBias(mpLastFrame->GetImuBias());
            PreintegrateIMU();
        }
        clock_t s_1_1, s_1_2, e_1_1, e_1_2;
        double cam_pos_time;
        s_1_1 = clock();
        // Get initial estimate using P3P plus RanSac
       // std::cout<<"last tcw: "<<mpLastFrame->mTcw<<std::endl;
        cv::Mat iniTcw = GetInitModelCam(TemperalMatch,TemperalMatch_subset);
        e_1_1 = clock();
        //std::cout<<"init: "<<iniTcw<<std::endl;
        s_1_2 = clock();
        // cout << "the ground truth pose: " << endl << mpCurrentFrame->mTcw_gt << endl;
        // cout << "initial pose: " << endl << iniTcw << endl;
        // // compute the pose with new matching
        mpCurrentFrame->SetPose(iniTcw);
        if (bJoint)
            Optimizer::PoseOptimizationFlow2Cam(mpCurrentFrame, mpLastFrame, TemperalMatch_subset);
        else
            Optimizer::PoseOptimizationNew(mpCurrentFrame, mpLastFrame, TemperalMatch_subset);
        e_1_2 = clock();
        cam_pos_time = (double)(e_1_1-s_1_1)/CLOCKS_PER_SEC*1000 + (double)(e_1_2-s_1_2)/CLOCKS_PER_SEC*1000;
        all_timing[1] = cam_pos_time;
        // cout << "camera pose estimation time: " << cam_pos_time << endl;

        // Update motion model
        if(!mpLastFrame->mTcw.empty())
        {
            cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
            mpLastFrame->GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
            mpLastFrame->GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
            mVelocity = mpCurrentFrame->mTcw*LastTwc;
        }


        // // **** show the picked points ****
        std::vector<cv::KeyPoint> PickKeys;
        for (int j = 0; j < TemperalMatch_subset.size(); ++j){
            PickKeys.push_back(mpCurrentFrame->mvStatKeys[TemperalMatch_subset[j]]);
        }
        //cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
       // cv::imshow("KeyPoints and Grid on Background", mImGray);
        //cv::waitKey(10);



        // // ====== compute sparse scene flow to the found matches =======
        GetSceneFlowObj();

        // // ---------------------------------------------------------------------------------------
        // // ++++++++++++++++++++++++++++++++ Dynamic Object Tracking ++++++++++++++++++++++++++++++
        // // ---------------------------------------------------------------------------------------

        std::vector<std::vector<int> > ObjIdNew = DynObjTracking();

        // // ---------------------------------------------------------------------------------------
        // // ++++++++++++++++++++++++++++++ Object Motion Estimation +++++++++++++++++++++++++++++++
        // // ---------------------------------------------------------------------------------------

        clock_t s_3_1, s_3_2, e_3_1, e_3_2;
        double obj_mot_time = 0, t_con = 0;
        mpCurrentFrame->bObjStat.resize(ObjIdNew.size(),true);
        mpCurrentFrame->vObjMod.resize(ObjIdNew.size());
        mpCurrentFrame->vObjPosePre.resize(ObjIdNew.size());
        mpCurrentFrame->vObjMod_gt.resize(ObjIdNew.size());
        mpCurrentFrame->vObjSpeed_gt.resize(ObjIdNew.size());
        mpCurrentFrame->vSpeed.resize(ObjIdNew.size());
        mpCurrentFrame->vObjBoxID.resize(ObjIdNew.size());
        mpCurrentFrame->vObjCentre3D.resize(ObjIdNew.size());
        mpCurrentFrame->vnObjID.resize(ObjIdNew.size());
        mpCurrentFrame->vnObjInlierID.resize(ObjIdNew.size());
        repro_e.resize(ObjIdNew.size(),0.0);
        // cv::Mat Last_Twc_gt = Converter::toInvMatrix(mpLastFrame->mTcw_gt);
        // cv::Mat Curr_Twc_gt = Converter::toInvMatrix(mpCurrentFrame->mTcw_gt);
        // main loop
        for (int i = 0; i < ObjIdNew.size(); ++i)
        {

            cv::Mat ObjCentre3D_pre = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
            for (int j = 0; j < ObjIdNew[i].size(); ++j)
            {
                // save object centroid in current frame
                cv::Mat x3D_p = mpLastFrame->UnprojectStereoObject(ObjIdNew[i][j],0);
                ObjCentre3D_pre = ObjCentre3D_pre + x3D_p;

            }
            ObjCentre3D_pre = ObjCentre3D_pre/ObjIdNew[i].size();
            mpCurrentFrame->vObjCentre3D[i] = ObjCentre3D_pre;


            s_3_1 = clock();

            // ******* Get initial model and inlier set using P3P RanSac ********
            std::vector<int> ObjIdTest = ObjIdNew[i];
            mpCurrentFrame->vnObjID[i] = ObjIdTest;
            std::vector<int> ObjIdTest_in;
            mpCurrentFrame->mInitModel = GetInitModelObj(ObjIdTest,ObjIdTest_in,i);
            // cv::Mat H_tmp = Converter::toInvMatrix(mpCurrentFrame->mTcw_gt)*mpCurrentFrame->mInitModel;
            // cout << "Initial motion estimation: " << endl << H_tmp << endl;
            e_3_1 = clock();

            if (ObjIdTest_in.size()<50)
            {
                mpCurrentFrame->bObjStat[i] = false;
                mpCurrentFrame->vObjMod_gt[i] = cv::Mat::eye(4,4, CV_32F);
                mpCurrentFrame->vObjMod[i] = cv::Mat::eye(4,4, CV_32F);
                mpCurrentFrame->vObjCentre3D[i] = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
                mpCurrentFrame->vObjSpeed_gt[i] = 0.0;
                mpCurrentFrame->vSpeed[i] = cv::Point2f(0.f, 0.f);
                mpCurrentFrame->vnObjInlierID[i] = ObjIdTest_in;
                continue;
            }

            // cout << "number of pick points: " << ObjIdTest_in.size() << "/" << ObjIdTest.size() << "/" << mpCurrentFrame->mvObjKeys.size() << endl;

            // // **** show the picked points ****
            // std::vector<cv::KeyPoint> PickKeys;
            // for (int j = 0; j < ObjIdTest_in.size(); ++j){
            //     // PickKeys.push_back(mpCurrentFrame->mvStatKeys[ObjIdTest[j]]);
            //     PickKeys.push_back(mpCurrentFrame->mvObjKeys[ObjIdTest_in[j]]);
            // }
            // cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
            // cv::imshow("KeyPoints and Grid on Vehicle", mImGray);
            // cv::waitKey(0);

            // // // // image show the matching on each object
            // std::vector<cv::KeyPoint> PreKeys, CurKeys;
            // std::vector<cv::DMatch> TMes;
            // for (int j = 0; j < ObjIdTest.size(); ++j)
            // {
            //     // save key points for visualization
            //     PreKeys.push_back(mpLastFrame->mvObjKeys[ObjIdTest[j]]);
            //     CurKeys.push_back(mpCurrentFrame->mvObjKeys[ObjIdTest[j]]);
            //     TMes.push_back(cv::DMatch(count,count,0));
            //     count = count + 1;
            // }
            // cout << "count count: " << count << endl;
            // cv::Mat img_matches;
            // drawMatches(mImGrayLast, PreKeys, mImGray, CurKeys,
            //             TMes, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            //             vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/1.0, img_matches.rows/1.0));
            // cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
            // cv::imshow("temperal matches", img_matches);
            // cv::waitKey(0);

            // ***************************************************************************************

            s_3_2 = clock();
            // // save object motion and label
            std::vector<int> InlierID;
            if (bJoint)
            {
                cv::Mat Obj_X_tmp = Optimizer::PoseOptimizationFlow2(mpCurrentFrame,mpLastFrame,ObjIdTest_in,InlierID);
                mpCurrentFrame->vObjMod[i] = Converter::toInvMatrix(mpCurrentFrame->mTcw)*Obj_X_tmp;
            }
            else
                mpCurrentFrame->vObjMod[i] = Optimizer::PoseOptimizationObjMot(mpCurrentFrame,mpLastFrame,ObjIdTest_in,InlierID);
            e_3_2 = clock();
            t_con = t_con + 1;
            obj_mot_time = obj_mot_time + (double)(e_3_1-s_3_1)/CLOCKS_PER_SEC*1000 + (double)(e_3_2-s_3_2)/CLOCKS_PER_SEC*1000;

            mpCurrentFrame->vnObjInlierID[i] = InlierID;

            // cout << "computed motion of object No. " << mpCurrentFrame->nSemPosition[i] << " :" << endl;
            // cout << mpCurrentFrame->vObjMod[i] << endl;

            // ***********************************************************************************************

            // // ***** get the ground truth object speed here ***** (use version 1 here)
            // cv::Mat sp_gt_v, sp_gt_v2;
            // sp_gt_v = H_p_c.rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-H_p_c.rowRange(0,3).colRange(0,3))*ObjCentre3D_pre; // L_w_p.rowRange(0,3).col(3) or ObjCentre3D_pre
            // sp_gt_v2 = L_w_p.rowRange(0,3).col(3) - L_w_c.rowRange(0,3).col(3);
            // float sp_gt_norm = std::sqrt( sp_gt_v.at<float>(0)*sp_gt_v.at<float>(0) + sp_gt_v.at<float>(1)*sp_gt_v.at<float>(1) + sp_gt_v.at<float>(2)*sp_gt_v.at<float>(2) )*36;
            // // float sp_gt_norm2 = std::sqrt( sp_gt_v2.at<float>(0)*sp_gt_v2.at<float>(0) + sp_gt_v2.at<float>(1)*sp_gt_v2.at<float>(1) + sp_gt_v2.at<float>(2)*sp_gt_v2.at<float>(2) )*36;
            // mpCurrentFrame->vObjSpeed_gt[i] = sp_gt_norm;

            // // ***** calculate the estimated object speed *****
            cv::Mat sp_est_v;
            sp_est_v = mpCurrentFrame->vObjMod[i].rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-mpCurrentFrame->vObjMod[i].rowRange(0,3).colRange(0,3))*ObjCentre3D_pre;
            float sp_est_norm = std::sqrt( sp_est_v.at<float>(0)*sp_est_v.at<float>(0) + sp_est_v.at<float>(1)*sp_est_v.at<float>(1) + sp_est_v.at<float>(2)*sp_est_v.at<float>(2) )*36;

            // cout << "estimated and ground truth object speed: " << sp_est_norm << "km/h " << sp_gt_norm << "km/h " << endl;
            cout << "Dynamic object ID "<< i <<" estimated speed: " << sp_est_norm << "km/h " << endl;

            mpCurrentFrame->vSpeed[i].x = sp_est_norm*36;
            // mpCurrentFrame->vSpeed[i].y = sp_gt_norm*36;

        }

        if (t_con!=0)
        {
            obj_mot_time = obj_mot_time/t_con;
            all_timing[3] = obj_mot_time;
            // cout << "object motion estimation time: " << obj_mot_time << endl;
        }
        else
            all_timing[3] = 0;

        // ****** Renew Current frame information *******

        clock_t s_4, e_4;
        double map_upd_time;
        s_4 = clock();
        RenewFrameInfo(TemperalMatch_subset);
        e_4 = clock();
        map_upd_time = (double)(e_4-s_4)/CLOCKS_PER_SEC*1000;
        all_timing[4] = map_upd_time;
        // cout << "map updating time: " << map_upd_time << endl;

        // **********************************************

        // Save timing analysis to the map
        mpMap->vfAll_time.push_back(all_timing);


        // // ====== Update from current to last frames ======
        mvKeysLastFrame = mpLastFrame->mvStatKeys;  // new added (1st Dec)  mvStatKeys <-> mvKeys
        mvKeysCurrentFrame = mpCurrentFrame->mvStatKeys; // new added (12th Sep)
        mpCurrentFrame->mpPrevFrame = mpLastFrame;
        mpLastFrame->mpNextFrame = mpCurrentFrame;
        mpLastFrame = mpCurrentFrame; 
        mpLastFrame->mvStatKeys = mpCurrentFrame->mvStatKeysTmp; // new added Jul 30 2019
        mpLastFrame->mvStatDepth = mpCurrentFrame->mvStatDepthTmp;
        // **********************************************************
        // ********* save some stuffs for graph structure. **********
        // **********************************************************
        // (1) detected static features, corresponding depth and associations
        mpMap->vpFeatSta.push_back(mpCurrentFrame->mvStatKeysTmp);
        mpMap->vfDepSta.push_back(mpCurrentFrame->mvStatDepthTmp);
        mpMap->vp3DPointSta.push_back(mpCurrentFrame->mvStat3DPointTmp);  // (new added Dec 12 2019)
        mpMap->vnAssoSta.push_back(mpCurrentFrame->nStaInlierID);         // (new added Nov 14 2019)

        // (2) detected dynamic object features, corresponding depth and associations
        mpMap->vpFeatDyn.push_back(mpCurrentFrame->mvObjKeys);           // (new added Nov 20 2019)
        mpMap->vfDepDyn.push_back(mpCurrentFrame->mvObjDepth);           // (new added Nov 20 2019)
        mpMap->vp3DPointDyn.push_back(mpCurrentFrame->mvObj3DPoint);     // (new added Dec 12 2019)
        mpMap->vnAssoDyn.push_back(mpCurrentFrame->nDynInlierID);        // (new added Nov 20 2019)
        mpMap->vnFeatLabel.push_back(mpCurrentFrame->vObjLabel);         // (new added Nov 20 2019)

        // cout << "mpMap vpFfeatSta size " << mpMap->vpFeatSta.size() << endl;
        // cout << "mpMap vfDepSta size " << mpMap->vfDepSta.size() << endl;
        // cout << "mpMap vmCameraPose_GT size " << mpMap->vmCameraPose_GT.size() << endl;

        

        //Jesse -> this may be why when global batch is not running it doenst update the masks as well
        if (f_id==StopFrame || bLocalBatch)
        {
            // (3) save static feature tracklets
            mpMap->TrackletSta = GetStaticTrack();
            // (4) save dynamic feature tracklets
            mpMap->TrackletDyn = GetDynamicTrackNew();  // (new added Nov 20 2019)
        }


        // (5) camera pose
        cv::Mat CameraPoseTmp = Converter::toInvMatrix(mpCurrentFrame->mTcw);
        mpMap->vmCameraPose.push_back(CameraPoseTmp);
        mpMap->vmCameraPose_RF.push_back(CameraPoseTmp);
        // (6) Rigid motions and label, including camera (label=0) and objects (label>0)
        std::vector<cv::Mat> Mot_Tmp, ObjPose_Tmp;
        std::vector<int> Mot_Lab_Tmp, Sem_Lab_Tmp;
        std::vector<bool> Obj_Stat_Tmp;
        // (6.1) Save Camera Motion and Label
        cv::Mat CameraMotionTmp = Converter::toInvMatrix(mVelocity);
        Mot_Tmp.push_back(CameraMotionTmp);
        // ObjPose_Tmp.push_back(CameraMotionTmp); -> jesse comments this -> just looks like it shouldn't be here...?
        Mot_Lab_Tmp.push_back(0);
        Sem_Lab_Tmp.push_back(0);
        Obj_Stat_Tmp.push_back(true);
        // (6.2) Save Object Motions and Label
        for (int i = 0; i < mpCurrentFrame->vObjMod.size(); ++i)
        {
            if (!mpCurrentFrame->bObjStat[i])
                continue;
            Obj_Stat_Tmp.push_back(mpCurrentFrame->bObjStat[i]);
            Mot_Tmp.push_back(mpCurrentFrame->vObjMod[i]);
            ObjPose_Tmp.push_back(mpCurrentFrame->vObjPosePre[i]);
            Mot_Lab_Tmp.push_back(mpCurrentFrame->nModLabel[i]);
            Sem_Lab_Tmp.push_back(mpCurrentFrame->nSemPosition[i]);
        }
        // (6.3) Save to The Map
        mpMap->vmRigidMotion.push_back(Mot_Tmp);
        mpMap->vmObjPosePre.push_back(ObjPose_Tmp);
        mpMap->vmRigidMotion_RF.push_back(Mot_Tmp);
        mpMap->vnRMLabel.push_back(Mot_Lab_Tmp);
        mpMap->vnSMLabel.push_back(Sem_Lab_Tmp);
        mpMap->vbObjStat.push_back(Obj_Stat_Tmp);

        mpMap->AddFrame(mpCurrentFrame);
      
        // (10) Computed Camera and Object Speeds
        std::vector<cv::Mat> Centre_Tmp;
        // (10.1) Save Camera Speed
        cv::Mat CameraCentre = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
        Centre_Tmp.push_back(CameraCentre);
        // (10.2) Save Object Motions
        for (int i = 0; i < mpCurrentFrame->vObjCentre3D.size(); ++i)
        {
            if (!mpCurrentFrame->bObjStat[i])
                continue;
            Centre_Tmp.push_back(mpCurrentFrame->vObjCentre3D[i]);
        }
        // (10.3) Save to The Map
        mpMap->vmRigidCentre.push_back(Centre_Tmp);
    }

    // =================================================================================================
    // ============== Partial batch optimize on all the measurements (local optimization) ==============
    // =================================================================================================

    
    bLocalBatch = true;
    if ( bLocalBatch)
    {
        int window=nWINDOW_SIZE;
        if(f_id<nWINDOW_SIZE)
           window=f_id;
        clock_t s_5, e_5;
        double loc_ba_time;
        s_5 = clock();
        if(mSensor==System::IMU_RGBD && isImuInitialized())
        {
            //Optimizer::LocalInertialBA(mpMap,mK,nWINDOW_SIZE);
            Optimizer::PartialBatchOptimization(mpMap,mK,window);
        }
        else{
            // Get Partial Batch Optimization
            Optimizer::PartialBatchOptimization(mpMap,mK,window);
        }
        // std::cout<<"ba optimizer: "<<mpCurrentFrame->mTcw<<std::endl;
        e_5 = clock();
        loc_ba_time = (double)(e_5-s_5)/CLOCKS_PER_SEC*1000;
        mpMap->fLBA_time.push_back(loc_ba_time);
        if(mSensor==System::IMU_RGBD && !isImuInitialized())
           InitializeIMU(1e2, 1e9, true);
        
        if(mState==Tracking::OK && isImuInitialized() && mTinit<100.0f){
            mTinit += mpCurrentFrame->mTimeStamp - mpCurrentFrame->mpPrevFrame->mTimeStamp;
            // if(!mbIMU_BA1 && mTinit>3.0f){
            //     cout << "start VIBA 1" << endl;
            //     mbIMU_BA1=true;
            //     InitializeIMU(1.f, 1e5, true); // 1.f, 1e5
            //     cout << "end VIBA 1" << endl;
            // }else if(!mbIMU_BA2 && mTinit>10.0f){
            //     cout << "start VIBA 2" << endl;
            //     mbIMU_BA2=true;
            //     InitializeIMU(0.f, 0.f, true); // 0.f, 0.f
            //     cout << "end VIBA 2" << endl;
            // }
            if (((mpMap->GetFramesInMapSize())<=1000) &&
                  ((mTinit>15.0f && mTinit<15.5f)||
                  (mTinit>25.0f && mTinit<25.5f)||
                  (mTinit>35.0f && mTinit<35.5f)||
                  (mTinit>45.0f && mTinit<45.5f)||
                  (mTinit>55.0f && mTinit<55.5f)||
                  (mTinit>65.0f && mTinit<65.5f)||
                  (mTinit>75.0f && mTinit<75.5f))){
                      cout << "start scale ref" << endl;
                      ScaleRefinement();
                      cout << "end scale ref" << endl;
                   }
        }
    
        // cout << "local optimization time: " << loc_ba_time << endl;
    }

    // =================================================================================================
    // ============== Full batch optimize on all the measurements (global optimization) ================
    // =================================================================================================

    bGlobalBatch = true;
    if (f_id==StopFrame) // bFrame2Frame f_id>=2
    {
        cout << "Fid is stop frame: " << f_id << endl;
        if (bGlobalBatch && mTestData==KITTI)
        {
            // Get Full Batch Optimization
            Optimizer::FullBatchOptimization(mpMap,mK);
            f_id = 0; 
        }
        else {
            mState = OK;
        }

    }
    else {
        mState = OK;

    }

}


void Tracking::Initialization()
{

    // initialize the 3d points
    {
        // static
        std::vector<cv::Mat> mv3DPointTmp;
        for (int i = 0; i < mpCurrentFrame->mvStatKeysTmp.size(); ++i)
        {
            mv3DPointTmp.push_back(Optimizer::Get3DinCamera(mpCurrentFrame->mvStatKeysTmp[i], mpCurrentFrame->mvStatDepthTmp[i], mK));
        }
        mpCurrentFrame->mvStat3DPointTmp = mv3DPointTmp;
        // dynamic
        std::vector<cv::Mat> mvObj3DPointTmp;
        for (int i = 0; i < mpCurrentFrame->mvObjKeys.size(); ++i)
        {
            mvObj3DPointTmp.push_back(Optimizer::Get3DinCamera(mpCurrentFrame->mvObjKeys[i], mpCurrentFrame->mvObjDepth[i], mK));
        }
        mpCurrentFrame->mvObj3DPoint = mvObj3DPointTmp;
        // cout << "see the size 1: " << mpCurrentFrame->mvStatKeysTmp.size() << " " << mpCurrentFrame->mvSift3DPoint.size() << endl;
        // cout << "see the size 2: " << mpCurrentFrame->mvObjKeys.size() << " " << mpCurrentFrame->mvObj3DPoint.size() << endl;
    }

    // (1) save detected static features and corresponding depth
    mpMap->vpFeatSta.push_back(mpCurrentFrame->mvStatKeysTmp);  // modified Nov 14 2019
    mpMap->vfDepSta.push_back(mpCurrentFrame->mvStatDepthTmp);  // modified Nov 14 2019
    mpMap->vp3DPointSta.push_back(mpCurrentFrame->mvStat3DPointTmp);  // modified Dec 17 2019
    // (2) save detected dynamic object features and corresponding depth
    mpMap->vpFeatDyn.push_back(mpCurrentFrame->mvObjKeys);  // modified Nov 19 2019
    mpMap->vfDepDyn.push_back(mpCurrentFrame->mvObjDepth);  // modified Nov 19 2019
    mpMap->vp3DPointDyn.push_back(mpCurrentFrame->mvObj3DPoint);  // modified Dec 17 2019
    // (3) save camera pose
    mpMap->vmCameraPose.push_back(cv::Mat::eye(4,4,CV_32F));
    mpMap->vmCameraPose_RF.push_back(cv::Mat::eye(4,4,CV_32F));
    mpMap->vmCameraPose_GT.push_back(cv::Mat::eye(4,4,CV_32F));

    // cout << "mpMap vpFfeatSta size " << mpMap->vpFeatSta.size() << endl;
    // cout << "mpMap vfDepSta size " << mpMap->vfDepSta.size() << endl;
    // cout << "mpMap vmCameraPose_GT size " << mpMap->vmCameraPose_GT.size() << endl;

    // cout << "mpCurrentFrame->N: " << mpCurrentFrame->N << endl;

    // Set Frame pose to the origin
    if (mSensor == System::IMU_RGBD)
        {
            cv::Mat Rwb0 = mpCurrentFrame->mImuCalib.Tcb.rowRange(0,3).colRange(0,3).clone();
            cv::Mat twb0 = mpCurrentFrame->mImuCalib.Tcb.rowRange(0,3).col(3).clone();
            mpCurrentFrame->SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3,1,CV_32F));
            mpCurrentFrame->mpImuPreintegrated = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
        }
        else
            mpCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));
    mpCurrentFrame->mTcw_gt = cv::Mat::eye(4,4,CV_32F);
    // mpCurrentFrame->mTcw_gt = Converter::toInvMatrix(mOriginInv)*mpCurrentFrame->mTcw_gt;
    // cout << "mTcw_gt: " << mpCurrentFrame->mTcw_gt << endl;
    // bFirstFrame = false;
    // cout << "current pose: " << endl << mpCurrentFrame->mTcw_gt << endl;
    // cout << "current pose inverse: " << endl << mOriginInv << endl;

    mpLastFrame =mpCurrentFrame;  //  important !!!
    mpLastFrame->mvStatKeys = mpCurrentFrame->mvStatKeysTmp; // new added Jul 30 2019
    mpLastFrame->mvStatDepth = mpCurrentFrame->mvStatDepthTmp;  // new added Jul 30 2019
    mpLastFrame->N_s = mpCurrentFrame->N_s_tmp;
    mvKeysLastFrame = mpLastFrame->mvStatKeys; // +++ new added +++
    
    mState=OK;

    cout << "Initialization, Done!" << endl;
}

void Tracking::GetSceneFlowObj()
{
    // // Threshold // //
    // int max_dist = 90, max_lat = 30;
    // double fps = 10, max_velocity_ms = 40;
    // double max_depth = 30;

    // Initialization
    int N = mpCurrentFrame->mvObjKeys.size();
    mpCurrentFrame->vFlow_3d.resize(N);
    // mpCurrentFrame->vFlow_2d.resize(N);

    std::vector<Eigen::Vector3d> pts_p3d(N,Eigen::Vector3d(-1,-1,-1)), pts_vel(N,Eigen::Vector3d(-1,-1,-1));

    const cv::Mat Rcw = mpCurrentFrame->mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = mpCurrentFrame->mTcw.rowRange(0,3).col(3);

    // Main loop
    for (int i = 0; i < N; ++i)
    {
        // // filter
        // if(mpCurrentFrame->mvObjDepth[i]>max_depth  || mpLastFrame->mvObjDepth[i]>max_depth)
        // {
        //     mpCurrentFrame->vObjLabel[i]=-1;
        //     continue;
        // }
        if (mpCurrentFrame->vSemObjLabel[i]<=0 || mpLastFrame->vSemObjLabel[i]<=0)
        {
            mpCurrentFrame->vObjLabel[i]=-1;
            continue;
        }

        // get the 3d flow
        cv::Mat x3D_p = mpLastFrame->UnprojectStereoObject(i,0);
        cv::Mat x3D_c = mpCurrentFrame->UnprojectStereoObject(i,0);

        pts_p3d[i] << x3D_p.at<float>(0), x3D_p.at<float>(1), x3D_p.at<float>(2);

        // cout << "3d points: " << x3D_p << " " << x3D_c << endl;

        cv::Point3f flow3d;
        flow3d.x = x3D_c.at<float>(0) - x3D_p.at<float>(0);
        flow3d.y = x3D_c.at<float>(1) - x3D_p.at<float>(1);
        flow3d.z = x3D_c.at<float>(2) - x3D_p.at<float>(2);

        pts_vel[i] << flow3d.x, flow3d.y, flow3d.z;

        // cout << "3d points: " << mpCurrentFrame->vFlow_3d[i] << endl;

        // // threshold the velocity
        // if(cv::norm(flow3d)*fps > max_velocity_ms)
        // {
        //     mpCurrentFrame->vObjLabel[i]=-1;
        //     continue;
        // }

        mpCurrentFrame->vFlow_3d[i] = flow3d;

        // // get the 2D re-projection error vector
        // // (1) transfer 3d from world to current frame.
        // cv::Mat x3D_pc = Rcw*x3D_p+tcw;
        // // (2) project 3d into current image plane
        // float xc = x3D_pc.at<float>(0);
        // float yc = x3D_pc.at<float>(1);
        // float invzc = 1.0/x3D_pc.at<float>(2);
        // float u = mpCurrentFrame->fx*xc*invzc+mpCurrentFrame->cx;
        // float v = mpCurrentFrame->fy*yc*invzc+mpCurrentFrame->cy;

        // mpCurrentFrame->vFlow_2d[i].x = mpCurrentFrame->mvObjKeys[i].pt.x - u;
        // mpCurrentFrame->vFlow_2d[i].y = mpCurrentFrame->mvObjKeys[i].pt.y - v;

        // // cout << "2d errors: " << mpCurrentFrame->vFlow_2d[i] << endl;

    }

    // // // ===== show scene flow from bird eye view =====
     cv::Mat img_sparse_flow_3d;
     BirdEyeVizProperties viz_props;
     viz_props.birdeye_scale_factor_ = 20.0;
     viz_props.birdeye_left_plane_ = -15.0;
     viz_props.birdeye_right_plane_ = 15.0;
     viz_props.birdeye_far_plane_ = 30.0;

    //  Tracking::DrawSparseFlowBirdeye(pts_p3d, pts_vel, Converter::toInvMatrix(mpLastFrame->mTcw), viz_props, img_sparse_flow_3d);
    //  cv::imshow("SparseFlowBirdeye", img_sparse_flow_3d*255);
    //  cv::waitKey(0);
}

std::vector<std::vector<int> > Tracking::DynObjTracking()
{
    clock_t s_2, e_2;
    double obj_tra_time;
    s_2 = clock();

    // Find the unique labels in semantic label
    auto UniLab = mpCurrentFrame->vSemObjLabel;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );

    // cout << "Unique Semantic Label: ";
    // for (int i = 0; i < UniLab.size(); ++i)
    //     cout  << UniLab[i] << " ";
    // cout << endl;

    // Collect the predicted labels and semantic labels in vector
    std::vector<std::vector<int> > Posi(UniLab.size());
    for (int i = 0; i < mpCurrentFrame->vSemObjLabel.size(); ++i)
    {
        // skip outliers
        if (mpCurrentFrame->vObjLabel[i]==-1)
            continue;

        // save object label
        for (int j = 0; j < UniLab.size(); ++j)
        {
            if(mpCurrentFrame->vSemObjLabel[i]==UniLab[j]){
                Posi[j].push_back(i);
                break;
            }
        }
    }

    // // Save objects only from Posi() -> ObjId()
    std::vector<std::vector<int> > ObjId;
    std::vector<int> sem_posi; // semantic label position for the objects
    int shrin_thr_row=10, shrin_thr_col=20;

    for (int i = 0; i < Posi.size(); ++i)
    {
        // shrink the image to get rid of object parts on the boundary
        float count = 0, count_thres=0.5;
        for (int j = 0; j < Posi[i].size(); ++j)
        {
            const float u = mpCurrentFrame->mvObjKeys[Posi[i][j]].pt.x;
            const float v = mpCurrentFrame->mvObjKeys[Posi[i][j]].pt.y;
            if ( v<shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u<shrin_thr_col || u>(mImGray.cols-shrin_thr_col) )
                count = count + 1;
        }
        if (count/Posi[i].size()>count_thres)
        {
            // cout << "Most part of this object is on the image boundary......" << endl;
            for (int k = 0; k < Posi[i].size(); ++k)
                mpCurrentFrame->vObjLabel[Posi[i][k]] = -1;
            continue;
        }
        else
        {
            ObjId.push_back(Posi[i]);
            sem_posi.push_back(UniLab[i]);
        }
    }

    // // Check scene flow distribution of each object and keep the dynamic object
    std::vector<std::vector<int> > ObjIdNew;
    std::vector<int> SemPosNew, obj_dis_tres(sem_posi.size(),0);
    for (int i = 0; i < ObjId.size(); ++i)
    {

        float obj_center_depth = 0, sf_min=100, sf_max=0, sf_mean=0, sf_count=0;
        std::vector<int> sf_range(10,0);
        for (int j = 0; j < ObjId[i].size(); ++j)
        {
            obj_center_depth = obj_center_depth + mpCurrentFrame->mvObjDepth[ObjId[i][j]];
            // const float sf_norm = cv::norm(mpCurrentFrame->vFlow_3d[ObjId[i][j]]);
            float sf_norm = std::sqrt(mpCurrentFrame->vFlow_3d[ObjId[i][j]].x*mpCurrentFrame->vFlow_3d[ObjId[i][j]].x + mpCurrentFrame->vFlow_3d[ObjId[i][j]].z*mpCurrentFrame->vFlow_3d[ObjId[i][j]].z);
            if (sf_norm<fSFMgThres)
                sf_count = sf_count+1;
            if(sf_norm<sf_min)
                sf_min = sf_norm;
            if(sf_norm>sf_max)
                sf_max = sf_norm;
            sf_mean = sf_mean + sf_norm;
            {
                if (0.0<=sf_norm && sf_norm<0.05)
                    sf_range[0] = sf_range[0] + 1;
                else if (0.05<=sf_norm && sf_norm<0.1)
                    sf_range[1] = sf_range[1] + 1;
                else if (0.1<=sf_norm && sf_norm<0.2)
                    sf_range[2] = sf_range[2] + 1;
                else if (0.2<=sf_norm && sf_norm<0.4)
                    sf_range[3] = sf_range[3] + 1;
                else if (0.4<=sf_norm && sf_norm<0.8)
                    sf_range[4] = sf_range[4] + 1;
                else if (0.8<=sf_norm && sf_norm<1.6)
                    sf_range[5] = sf_range[5] + 1;
                else if (1.6<=sf_norm && sf_norm<3.2)
                    sf_range[6] = sf_range[6] + 1;
                else if (3.2<=sf_norm && sf_norm<6.4)
                    sf_range[7] = sf_range[7] + 1;
                else if (6.4<=sf_norm && sf_norm<12.8)
                    sf_range[8] = sf_range[8] + 1;
                else if (12.8<=sf_norm && sf_norm<25.6)
                    sf_range[9] = sf_range[9] + 1;
            }
        }

        //cout << "scene flow distribution:"  << endl;
        // for (int j = 0; j < sf_range.size(); ++j)
        //     cout << sf_range[j] << " ";
        // cout << endl;

        if (sf_count/ObjId[i].size()>fSFDsThres)
        {
            // label this object as static background
            for (int k = 0; k < ObjId[i].size(); ++k)
                mpCurrentFrame->vObjLabel[ObjId[i][k]] = 0;
            continue;
        }
        else if (obj_center_depth/ObjId[i].size()>mThDepthObj || ObjId[i].size()<150)
        {
            obj_dis_tres[i]=-1;
            // cout << "object " << sem_posi[i] <<" is too far away or too small! " << obj_center_depth/ObjId[i].size() << endl;
            // label this object as far away object
            for (int k = 0; k < ObjId[i].size(); ++k)
                mpCurrentFrame->vObjLabel[ObjId[i][k]] = -1;
            continue;
        }
        else
        {
            // cout << "get new objects!" << endl;
            ObjIdNew.push_back(ObjId[i]);
            SemPosNew.push_back(sem_posi[i]);
        }
    }

    // add ground truth tracks
    std::vector<int> nSemPosi_gt_tmp = mpCurrentFrame->nSemPosi_gt;
    for (int i = 0; i < sem_posi.size(); ++i)
    {
        for (int j = 0; j < nSemPosi_gt_tmp.size(); ++j)
        {
            if (sem_posi[i]==nSemPosi_gt_tmp[j] && obj_dis_tres[i]==-1)
            {
                nSemPosi_gt_tmp[j]=-1;
            }
        }
    }

    mpMap->vnSMLabelGT.push_back(nSemPosi_gt_tmp);


    // // *** show the points on object ***
    // for (int i = 0; i < ObjIdNew.size(); ++i)
    // {
    //     // **** show the picked points ****
    //     std::vector<cv::KeyPoint> PickKeys;
    //     for (int j = 0; j < ObjIdNew[i].size(); ++j){
    //         PickKeys.push_back(mpCurrentFrame->mvObjKeys[ObjIdNew[i][j]]);
    //     }
    //     cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    //     cv::imshow("KeyPoints and Grid on Background", mImGray);
    //     cv::waitKey(0);
    // }

    // Relabel the objects that associate with the objects in last frame

    // initialize global object id
    if (f_id==1)
        max_id = 1;

    // save current label id
    std::vector<int> LabId(ObjIdNew.size());
    for (int i = 0; i < ObjIdNew.size(); ++i)
    {
        // save semantic labels in last frame
        std::vector<int> Lb_last;
        for (int k = 0; k < ObjIdNew[i].size(); ++k)
            Lb_last.push_back(mpLastFrame->vSemObjLabel[ObjIdNew[i][k]]);

        // find label that appears most in Lb_last()
        // (1) count duplicates
        std::map<int, int> dups;
        for(int k : Lb_last)
            ++dups[k];
        // (2) and sort them by descending order
        std::vector<std::pair<int, int> > sorted;
        for (auto k : dups)
            sorted.push_back(std::make_pair(k.first,k.second));
        std::sort(sorted.begin(), sorted.end(), SortPairInt);

        // label the object in current frame
        int New_lab = sorted[0].first;
        // cout << " what is in the new label: " << New_lab << endl;
        if (max_id==1)
        {
            LabId[i] = max_id;
            for (int k = 0; k < ObjIdNew[i].size(); ++k)
                mpCurrentFrame->vObjLabel[ObjIdNew[i][k]] = max_id;
            max_id = max_id + 1;
        }
        else
        {
            bool exist = false;
            for (int k = 0; k < mpLastFrame->nSemPosition.size(); ++k)
            {
                if (mpLastFrame->nSemPosition[k]==New_lab && mpLastFrame->bObjStat[k])
                {
                    LabId[i] = mpLastFrame->nModLabel[k];
                    for (int k = 0; k < ObjIdNew[i].size(); ++k)
                        mpCurrentFrame->vObjLabel[ObjIdNew[i][k]] = LabId[i];
                    exist = true;
                    break;
                }
            }
            if (exist==false)
            {
                LabId[i] = max_id;
                for (int k = 0; k < ObjIdNew[i].size(); ++k)
                    mpCurrentFrame->vObjLabel[ObjIdNew[i][k]] = max_id;
                max_id = max_id + 1;
            }
        }

    }

    // // assign the model label in current frame
    mpCurrentFrame->nModLabel = LabId;
    mpCurrentFrame->nSemPosition = SemPosNew;

    e_2 = clock();
    obj_tra_time = (double)(e_2-s_2)/CLOCKS_PER_SEC*1000;
    all_timing[2] = obj_tra_time;
    // cout << "dynamic object tracking time: " << obj_tra_time << endl;

    // cout << "Current Max_id: ("<< max_id << ") motion label: ";
    // for (int i = 0; i < LabId.size(); ++i)
    //     cout <<  LabId[i] << " ";


    return ObjIdNew;
}

cv::Mat Tracking::GetInitModelCam(const std::vector<int> &MatchId, std::vector<int> &MatchId_sub)
{
    cv::Mat Mod = cv::Mat::eye(4,4,CV_32F);
    int N = MatchId.size();

    // construct input
    std::vector<cv::Point2f> cur_2d(N);
    std::vector<cv::Point3f> pre_3d(N);
    std::vector<int>outliners(N,0);
    for (int i = 0; i < N; ++i)
    {
        cv::Point2f tmp_2d;
        tmp_2d.x = mpCurrentFrame->mvStatKeys[MatchId[i]].pt.x;
        tmp_2d.y = mpCurrentFrame->mvStatKeys[MatchId[i]].pt.y;
        cur_2d[i] = tmp_2d;
        cv::Point3f tmp_3d;
        cv::Mat x3D_p = mpLastFrame->UnprojectStereoStat(MatchId[i],0);
        if(x3D_p.empty()) {
            outliners[i]=-1;
            continue;
        }
        //std::cout<<"x3d : "<<x3D_p<<std::endl;
        tmp_3d.x = x3D_p.at<float>(0);
        tmp_3d.y = x3D_p.at<float>(1);
        tmp_3d.z = x3D_p.at<float>(2);
        pre_3d[i] = tmp_3d;
    }
    std::vector<cv::Point2f> good_cur_2d;
    std::vector<cv::Point3f> good_pre_3d;
    for(int i=0;i<N;++i){
        if(outliners[i]==0){
            good_cur_2d.push_back(cur_2d[i]);
            good_pre_3d.push_back(pre_3d[i]);
        }
    }
    // camera matrix & distortion coefficients
    cv::Mat camera_mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
    camera_mat.at<double>(0, 0) = mK.at<float>(0,0);
    camera_mat.at<double>(1, 1) = mK.at<float>(1,1);
    camera_mat.at<double>(0, 2) = mK.at<float>(0,2);
    camera_mat.at<double>(1, 2) = mK.at<float>(1,2);
    camera_mat.at<double>(2, 2) = 1.0;

    // output
    cv::Mat Rvec(3, 1, CV_64FC1);
    cv::Mat Tvec(3, 1, CV_64FC1);
    cv::Mat d(3, 3, CV_64FC1);
    cv::Mat inliers;

    // solve
    int iter_num = 500;
    double reprojectionError = 0.4, confidence = 0.98; // 0.5 0.3
    cv::solvePnPRansac(good_pre_3d, good_cur_2d, camera_mat, distCoeffs, Rvec, Tvec, false,
               iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_P3P); // AP3P EPNP P3P ITERATIVE DLS

    cv::Rodrigues(Rvec, d);

    // assign the result to current pose
    Mod.at<float>(0,0) = d.at<double>(0,0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0,2) = d.at<double>(0,2); Mod.at<float>(0,3) = Tvec.at<double>(0,0);
    Mod.at<float>(1,0) = d.at<double>(1,0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1,2) = d.at<double>(1,2); Mod.at<float>(1,3) = Tvec.at<double>(1,0);
    Mod.at<float>(2,0) = d.at<double>(2,0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2,2) = d.at<double>(2,2); Mod.at<float>(2,3) = Tvec.at<double>(2,0);


    // calculate the re-projection error
    std::vector<int> MM_inlier;
    cv::Mat MotionModel;
    if (mVelocity.empty())
        MotionModel = cv::Mat::eye(4,4,CV_32F)*mpLastFrame->mTcw;
    else
        MotionModel = mVelocity*mpLastFrame->mTcw;
    for (int i = 0; i < N; ++i)
    {
        const cv::Mat x3D  = (cv::Mat_<float>(3,1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
        const cv::Mat x3D_c = MotionModel.rowRange(0,3).colRange(0,3)*x3D+MotionModel.rowRange(0,3).col(3);

        const float xc = x3D_c.at<float>(0);
        const float yc = x3D_c.at<float>(1);
        const float invzc = 1.0/x3D_c.at<float>(2);
        const float u = mpCurrentFrame->fx*xc*invzc+mpCurrentFrame->cx;
        const float v = mpCurrentFrame->fy*yc*invzc+mpCurrentFrame->cy;
        const float u_ = cur_2d[i].x - u;
        const float v_ = cur_2d[i].y - v;
        const float Rpe = std::sqrt(u_*u_ + v_*v_);
        if (Rpe<reprojectionError){
            MM_inlier.push_back(i);
        }
    }

    // cout << "Inlier Compare: " << "(1)AP3P RANSAC: " << inliers.rows << " (2)Motion Model: " << MM_inlier.size() << endl;

    cv::Mat output;

    if (inliers.rows>MM_inlier.size())
    {
        // save the inliers IDs
        output = Mod;
        MatchId_sub.resize(inliers.rows);
        for (int i = 0; i < MatchId_sub.size(); ++i){
            MatchId_sub[i] = MatchId[inliers.at<int>(i)];
        }
        //cout << "(Camera) AP3P+RanSac inliers/total number: " << inliers.rows << "/" << MatchId.size() << endl;
    }
    else
    {
        output = MotionModel;
        MatchId_sub.resize(MM_inlier.size());
        for (int i = 0; i < MatchId_sub.size(); ++i){
            MatchId_sub[i] = MatchId[MM_inlier[i]];
        }
        //cout << "(Camera) Motion Model inliers/total number: " << MM_inlier.size() << "/" << MatchId.size() << endl;
    }

    return output;
}

cv::Mat Tracking::GetInitModelObj(const std::vector<int> &ObjId, std::vector<int> &ObjId_sub, const int objid)
{
    cv::Mat Mod = cv::Mat::eye(4,4,CV_32F);
    int N = ObjId.size();

    // construct input
    std::vector<cv::Point2f> cur_2d(N);
    std::vector<cv::Point3f> pre_3d(N);
    for (int i = 0; i < N; ++i)
    {
        cv::Point2f tmp_2d;
        tmp_2d.x = mpCurrentFrame->mvObjKeys[ObjId[i]].pt.x;
        tmp_2d.y = mpCurrentFrame->mvObjKeys[ObjId[i]].pt.y;
        cur_2d[i] = tmp_2d;
        cv::Point3f tmp_3d;
        cv::Mat x3D_p = mpLastFrame->UnprojectStereoObject(ObjId[i],0);
        tmp_3d.x = x3D_p.at<float>(0);
        tmp_3d.y = x3D_p.at<float>(1);
        tmp_3d.z = x3D_p.at<float>(2);
        pre_3d[i] = tmp_3d;
    }

    // camera matrix & distortion coefficients
    cv::Mat camera_mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
    camera_mat.at<double>(0, 0) = mK.at<float>(0,0);
    camera_mat.at<double>(1, 1) = mK.at<float>(1,1);
    camera_mat.at<double>(0, 2) = mK.at<float>(0,2);
    camera_mat.at<double>(1, 2) = mK.at<float>(1,2);
    camera_mat.at<double>(2, 2) = 1.0;

    // output
    cv::Mat Rvec(3, 1, CV_64FC1);
    cv::Mat Tvec(3, 1, CV_64FC1);
    cv::Mat d(3, 3, CV_64FC1);
    cv::Mat inliers;

    // solve
    int iter_num = 500;
    double reprojectionError = 0.4, confidence = 0.98; // 0.3 0.5 1.0
    cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, distCoeffs, Rvec, Tvec, false,
               iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_P3P); // AP3P EPNP P3P ITERATIVE DLS

    cv::Rodrigues(Rvec, d);

    // assign the result to current pose
    Mod.at<float>(0,0) = d.at<double>(0,0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0,2) = d.at<double>(0,2); Mod.at<float>(0,3) = Tvec.at<double>(0,0);
    Mod.at<float>(1,0) = d.at<double>(1,0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1,2) = d.at<double>(1,2); Mod.at<float>(1,3) = Tvec.at<double>(1,0);
    Mod.at<float>(2,0) = d.at<double>(2,0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2,2) = d.at<double>(2,2); Mod.at<float>(2,3) = Tvec.at<double>(2,0);

    // ******* Generate Motion Model if it does exist from previous frame *******
    int CurObjLab = mpCurrentFrame->nModLabel[objid];
    int PreObjID = -1;
    for (int i = 0; i < mpLastFrame->nModLabel.size(); ++i)
    {
        if (mpLastFrame->nModLabel[i]==CurObjLab)
        {
            PreObjID = i;
            break;
        }
    }

    cv::Mat MotionModel, output;
    std::vector<int> ObjId_tmp(N,-1); // new added Nov 19, 2019
    if (PreObjID!=-1)
    {
        // calculate the re-projection error
        std::vector<int> MM_inlier;
        MotionModel = mpCurrentFrame->mTcw*mpLastFrame->vObjMod[PreObjID];
        for (int i = 0; i < N; ++i)
        {
            const cv::Mat x3D  = (cv::Mat_<float>(3,1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
            const cv::Mat x3D_c = MotionModel.rowRange(0,3).colRange(0,3)*x3D+MotionModel.rowRange(0,3).col(3);

            const float xc = x3D_c.at<float>(0);
            const float yc = x3D_c.at<float>(1);
            const float invzc = 1.0/x3D_c.at<float>(2);
            const float u = mpCurrentFrame->fx*xc*invzc+mpCurrentFrame->cx;
            const float v = mpCurrentFrame->fy*yc*invzc+mpCurrentFrame->cy;
            const float u_ = cur_2d[i].x - u;
            const float v_ = cur_2d[i].y - v;
            const float Rpe = std::sqrt(u_*u_ + v_*v_);
            if (Rpe<reprojectionError){
                MM_inlier.push_back(i);
            }
        }

        // cout << "Inlier Compare: " << "(1)AP3P RANSAC: " << inliers.rows << " (2)Motion Model: " << MM_inlier.size() << endl;

        // ===== decide which model is best now =====
        if (inliers.rows>MM_inlier.size())
        {
            // save the inliers IDs
            output = Mod;
            ObjId_sub.resize(inliers.rows);
            for (int i = 0; i < ObjId_sub.size(); ++i){
                ObjId_sub[i] = ObjId[inliers.at<int>(i)];
                ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
            }
            // cout << "(Object) AP3P+RanSac inliers/total number: " << inliers.rows << "/" << ObjId.size() << endl;
        }
        else
        {
            output = MotionModel;
            ObjId_sub.resize(MM_inlier.size());
            for (int i = 0; i < ObjId_sub.size(); ++i){
                ObjId_sub[i] = ObjId[MM_inlier[i]];
                ObjId_tmp[MM_inlier[i]] = ObjId[MM_inlier[i]];
            }
            // cout << "(Object) Motion Model inliers/total number: " << MM_inlier.size() << "/" << ObjId.size() << endl;
        }
    }
    else
    {
        // save the inliers IDs
        output = Mod;
        ObjId_sub.resize(inliers.rows);
        for (int i = 0; i < ObjId_sub.size(); ++i){
            ObjId_sub[i] = ObjId[inliers.at<int>(i)];
            ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
        }
        // cout << "(Object) AP3P+RanSac [No MM] inliers/total number: " << inliers.rows << "/" << ObjId.size() << endl;
    }

    // update on vObjLabel (Nov 19 2019)
    for (int i = 0; i < ObjId_tmp.size(); ++i)
    {
        if (ObjId_tmp[i]==-1)
            mpCurrentFrame->vObjLabel[ObjId[i]]=-1;
    }

    return output;
}

void Tracking::DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar &color, int thickness, int line_type, const cv::Point2i &offset)
{

    auto cv_p1 = cv::Point2i(keys.pt.x,keys.pt.y);
    auto cv_p2 = cv::Point2i(keys.pt.x+flow.x,keys.pt.y+flow.y);
    //cout << "p1: " << cv_p1 << endl;
    //cout << "p2: " << cv_p2 << endl;

    bool p1_in_bounds = true;
    bool p2_in_bounds = true;
    if ((cv_p1.x < 0) && (cv_p1.y < 0) && (cv_p1.x > ref_image.cols) && (cv_p1.y > ref_image.rows) )
        p1_in_bounds = false;

    if ((cv_p2.x < 0) && (cv_p2.y < 0) && (cv_p2.x > ref_image.cols) && (cv_p2.y > ref_image.rows) )
        p2_in_bounds = false;

    // Draw line, but only if both end-points project into the image!
    if (p1_in_bounds || p2_in_bounds) { // This is correct. Won't draw only if both lines are out of bounds.
        // Draw line
        auto p1_offs = offset+cv_p1;
        auto p2_offs = offset+cv_p2;
        if (cv::clipLine(cv::Size(ref_image.cols, ref_image.rows), p1_offs, p2_offs)) {
            //cv::line(ref_image, p1_offs, p2_offs, color, thickness, line_type);
            cv::arrowedLine(ref_image, p1_offs, p2_offs, color, thickness, line_type);
        }
    }
}

void Tracking::DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image)
{
    for (int i=-radius; i<radius; i++) {
        for (int j=-radius; j<radius; j++) {
            int coord_y = center.y + i;
            int coord_x = center.x + j;

            if (coord_x>0 && coord_y>0 && coord_x<ref_image.cols && coord_y < ref_image.rows) {
                ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) = (1.0-alpha)*ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) + alpha*color;
            }
        }
    }
}

void Tracking::DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image)
{

    auto color = cv::Scalar(0.0, 0.0, 0.0);
    // Draw horizontal lines
    for (double i=0; i<viz_props.birdeye_far_plane_; i+=res_z) {
        double x_1 = viz_props.birdeye_left_plane_;
        double y_1 = i;
        double x_2 = viz_props.birdeye_right_plane_;
        double y_2 = i;
        TransformPointToScaledFrustum(x_1, y_1, viz_props);
        TransformPointToScaledFrustum(x_2, y_2, viz_props);
        auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
        cv::line(ref_image, p1, p2, color);
    }

    // Draw vertical lines
    for (double i=viz_props.birdeye_left_plane_; i<viz_props.birdeye_right_plane_; i+=res_x) {
        double x_1 = i;
        double y_1 = 0;
        double x_2 = i;
        double y_2 = viz_props.birdeye_far_plane_;
        TransformPointToScaledFrustum(x_1, y_1, viz_props);
        TransformPointToScaledFrustum(x_2, y_2, viz_props);
        auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
        cv::line(ref_image, p1, p2, color);
    }
}

void Tracking::DrawSparseFlowBirdeye(
        const std::vector<Eigen::Vector3d> &pts, const std::vector<Eigen::Vector3d> &vel,
        const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image)
{

    // For scaling / flipping cov. matrices
    Eigen::Matrix2d flip_mat;
    flip_mat << viz_props.birdeye_scale_factor_*1.0, 0, 0, viz_props.birdeye_scale_factor_*1.0;
    Eigen::Matrix2d world_to_cam_mat;
    const Eigen::Matrix4d &ref_to_rt_inv = Converter::toMatrix4d(camera);
    world_to_cam_mat << ref_to_rt_inv(0,0), ref_to_rt_inv(2,0), ref_to_rt_inv(0,2), ref_to_rt_inv(2,2);
    flip_mat = flip_mat*world_to_cam_mat;

    // Parameters
    // const int line_width = 2;

    ref_image = cv::Mat(viz_props.birdeye_scale_factor_*viz_props.birdeye_far_plane_,
                        (-viz_props.birdeye_left_plane_+viz_props.birdeye_right_plane_)*viz_props.birdeye_scale_factor_, CV_32FC3);
    ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));
    Tracking::DrawGridBirdeye(1.0, 1.0, viz_props, ref_image);


    for (int i=0; i<pts.size(); i++) {

        Eigen::Vector3d p_3d = pts[i];
        Eigen::Vector3d p_vel = vel[i];

        if (p_3d[0]==-1 || p_3d[1]==-1 || p_3d[2]<0)
            continue;
        if (p_vel[0]>0.1 || p_vel[2]>0.1)
            continue;

        // float xc = p_3d[0];
        // float yc = p_3d[1];
        // float invzc = 1.0/p_3d[2];
        // float u = mpCurrentFrame->fx*xc*invzc+mpCurrentFrame->cx;
        // float v = mpCurrentFrame->fy*yc*invzc+mpCurrentFrame->cy;
        // Eigen::Vector3i p_proj = Eigen::Vector3i(round(u), round(v), 1);
        const Eigen::Vector2d velocity = Eigen::Vector2d(p_vel[0], p_vel[2]); // !!!
        Eigen::Vector3d dir(velocity[0], 0.0, velocity[1]);

        double x_1 = p_3d[0];
        double z_1 = p_3d[2];

        double x_2 = x_1 + dir[0];
        double z_2 = z_1 + dir[2];

        // cout << dir[0] << " " << dir[2] << endl;

        if (x_1 > viz_props.birdeye_left_plane_ && x_2 > viz_props.birdeye_left_plane_ &&
            x_1 < viz_props.birdeye_right_plane_ && x_2 < viz_props.birdeye_right_plane_ &&
            z_1 > 0 && z_2 > 0 &&
            z_1 < viz_props.birdeye_far_plane_ && z_2 < viz_props.birdeye_far_plane_) {

            TransformPointToScaledFrustum(x_1, z_1, viz_props); //velocity[0], velocity[1]);
            TransformPointToScaledFrustum(x_2, z_2, viz_props); //velocity[0], velocity[1]);

            cv::arrowedLine(ref_image, cv::Point(x_1, z_1), cv::Point(x_2, z_2), cv::Scalar(1.0, 0.0, 0.0), 1);
            cv::circle(ref_image, cv::Point(x_1, z_1), 3.0, cv::Scalar(0.0, 0.0, 1.0), -1.0);
        }
    }

    // Coord. sys.
    int arrow_len = 60;
    int offset_y = 10;
    cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                    cv::Point(ref_image.cols/2+arrow_len, offset_y),
                    cv::Scalar(1.0, 0, 0), 2);
    cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                    cv::Point(ref_image.cols/2, offset_y+arrow_len),
                    cv::Scalar(0.0, 1.0, 0), 2);

    //cv::putText(ref_image, "X", cv::Point(ref_image.cols/2+arrow_len+10, offset_y+10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(1.0, 0, 0));
    //cv::putText(ref_image, "Z", cv::Point(ref_image.cols/2+10, offset_y+arrow_len), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0.0, 1.0, 0));

    // Flip image, because it is more intuitive to have ref. point at the bottom of the image
    cv::Mat dst;
    cv::flip(ref_image, dst, 0);
    ref_image = dst;
}

void Tracking::TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props)
{
    pose_x += (-viz_props.birdeye_left_plane_);
    pose_x *= viz_props.birdeye_scale_factor_;
    pose_z *= viz_props.birdeye_scale_factor_;
}

cv::Mat Tracking::ObjPoseParsingKT(const std::vector<float> &vObjPose_gt)
{
    // assign t vector
    cv::Mat t(3, 1, CV_32FC1);
    t.at<float>(0) = vObjPose_gt[6];
    t.at<float>(1) = vObjPose_gt[7];
    t.at<float>(2) = vObjPose_gt[8];

    // from Euler to Rotation Matrix
    cv::Mat R(3, 3, CV_32FC1);

    // assign r vector
    float y = vObjPose_gt[9]+(3.1415926/2); // +(3.1415926/2)
    float x = 0.0;
    float z = 0.0;

    // the angles are in radians.
    float cy = cos(y);
    float sy = sin(y);
    float cx = cos(x);
    float sx = sin(x);
    float cz = cos(z);
    float sz = sin(z);

    float m00, m01, m02, m10, m11, m12, m20, m21, m22;

    // ====== R = Ry*Rx*Rz =======

    // m00 = cy;
    // m01 = -sy;
    // m02 = 0;
    // m10 = sy;
    // m11 = cy;
    // m12 = 0;
    // m20 = 0;
    // m21 = 0;
    // m22 = 1;

    m00 = cy*cz+sy*sx*sz;
    m01 = -cy*sz+sy*sx*cz;
    m02 = sy*cx;
    m10 = cx*sz;
    m11 = cx*cz;
    m12 = -sx;
    m20 = -sy*cz+cy*sx*sz;
    m21 = sy*sz+cy*sx*cz;
    m22 = cy*cx;

    // ***************** old **************************

    // float alpha = vObjPose_gt[7]; // 7
    // float beta = vObjPose_gt[5]+(3.1415926/2);  // 5
    // float gamma = vObjPose_gt[6]; // 6

    // the angles are in radians.
    // float ca = cos(alpha);
    // float sa = sin(alpha);
    // float cb = cos(beta);
    // float sb = sin(beta);
    // float cg = cos(gamma);
    // float sg = sin(gamma);

    // float m00, m01, m02, m10, m11, m12, m20, m21, m22;

    // default
    // m00 = cb*ca;
    // m01 = cb*sa;
    // m02 = -sb;
    // m10 = sb*sg*ca-sa*cg;
    // m11 = sb*sg*sa+ca*cg;
    // m12 = cb*sg;
    // m20 = sb*cg*ca+sa*sg;
    // m21 = sb*cg*sa-ca*sg;
    // m22 = cb*cg;

    // m00 = ca*cb;
    // m01 = ca*sb*sg - sa*cg;
    // m02 = ca*sb*cg + sa*sg;
    // m10 = sa*cb;
    // m11 = sa*sb*sg + ca*cg;
    // m12 = sa*sb*cg - ca*sg;
    // m20 = -sb;
    // m21 = cb*sg;
    // m22 = cb*cg;

    // **************************************************

    R.at<float>(0,0) = m00;
    R.at<float>(0,1) = m01;
    R.at<float>(0,2) = m02;
    R.at<float>(1,0) = m10;
    R.at<float>(1,1) = m11;
    R.at<float>(1,2) = m12;
    R.at<float>(2,0) = m20;
    R.at<float>(2,1) = m21;
    R.at<float>(2,2) = m22;

    // construct 4x4 transformation matrix
    cv::Mat Pose = cv::Mat::eye(4,4,CV_32F);
    Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
    Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
    Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);

    // cout << "OBJ Pose: " << endl << Pose << endl;

    return Pose;

}

cv::Mat Tracking::ObjPoseParsingOX(const std::vector<float> &vObjPose_gt)
{
    // assign t vector
    cv::Mat t(3, 1, CV_32FC1);
    t.at<float>(0) = vObjPose_gt[2];
    t.at<float>(1) = vObjPose_gt[3];
    t.at<float>(2) = vObjPose_gt[4];

    // from axis-angle to Rotation Matrix
    cv::Mat R(3, 3, CV_32FC1);
    cv::Mat Rvec(3, 1, CV_32FC1);

    // assign r vector
    Rvec.at<float>(0,0) = vObjPose_gt[5];
    Rvec.at<float>(0,1) = vObjPose_gt[6];
    Rvec.at<float>(0,2) = vObjPose_gt[7];

    // *******************************************************************

    const float angle = std::sqrt(vObjPose_gt[5]*vObjPose_gt[5] + vObjPose_gt[6]*vObjPose_gt[6] + vObjPose_gt[7]*vObjPose_gt[7]);

    if (angle>0)
    {
        Rvec.at<float>(0,0) = Rvec.at<float>(0,0)/angle;
        Rvec.at<float>(0,1) = Rvec.at<float>(0,1)/angle;
        Rvec.at<float>(0,2) = Rvec.at<float>(0,2)/angle;
    }

    const float s = std::sin(angle);
    const float c = std::cos(angle);

    const float v = 1 - c;
    const float x = Rvec.at<float>(0,0);
    const float y = Rvec.at<float>(0,1);
    const float z = Rvec.at<float>(0,2);
    const float xyv = x*y*v;
    const float yzv = y*z*v;
    const float xzv = x*z*v;

    R.at<float>(0,0) = x*x*v + c;
    R.at<float>(0,1) = xyv - z*s;
    R.at<float>(0,2) = xzv + y*s;
    R.at<float>(1,0) = xyv + z*s;
    R.at<float>(1,1) = y*y*v + c;
    R.at<float>(1,2) = yzv - x*s;
    R.at<float>(2,0) = xzv - y*s;
    R.at<float>(2,1) = yzv + x*s;
    R.at<float>(2,2) = z*z*v + c;

    // ********************************************************************

    // cv::Rodrigues(Rvec, R);

    // construct 4x4 transformation matrix
    cv::Mat Pose = cv::Mat::eye(4,4,CV_32F);
    Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
    Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
    Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);

    // cout << "OBJ Pose: " << endl << Pose << endl;

    return Converter::toInvMatrix(mOriginInv)*Pose;

}


void Tracking::StackObjInfo(std::vector<cv::KeyPoint> &FeatDynObj, std::vector<float> &DepDynObj,
                  std::vector<int> &FeatLabObj)
{
    for (int i = 0; i < mpCurrentFrame->vnObjID.size(); ++i)
    {
        for (int j = 0; j < mpCurrentFrame->vnObjID[i].size(); ++j)
        {
            FeatDynObj.push_back(mpLastFrame->mvObjKeys[mpCurrentFrame->vnObjID[i][j]]);
            FeatDynObj.push_back(mpCurrentFrame->mvObjKeys[mpCurrentFrame->vnObjID[i][j]]);
            DepDynObj.push_back(mpLastFrame->mvObjDepth[mpCurrentFrame->vnObjID[i][j]]);
            DepDynObj.push_back(mpCurrentFrame->mvObjDepth[mpCurrentFrame->vnObjID[i][j]]);
            FeatLabObj.push_back(mpCurrentFrame->vObjLabel[mpCurrentFrame->vnObjID[i][j]]);
        }
    }
}

std::vector<std::vector<std::pair<int, int> > > Tracking::GetStaticTrack()
{
    // Get temporal match from Map
    std::vector<std::vector<int> > TemporalMatch = mpMap->vnAssoSta;
    int N = TemporalMatch.size();
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;
    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;

    // main loop
    int IDsofar = 0;
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(TemporalMatch[i].size(),-1);

        // check each feature
        for (int j = 0; j < TemporalMatch[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID; pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);

                    // save tracklet ID
                    TrackCheck_cur[j] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
                else
                    continue;
            }
            // frame i and i+1 (i>0)
            else
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    if (TrackCheck_pre[TemporalMatch[i][j]]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);

                        // save tracklet ID
                        TrackCheck_cur[j] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
                else
                    continue;
            }
        }

        TrackCheck_pre = TrackCheck_cur;
    }

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    // for (int i = 0; i < N; ++i)
    //     cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    // cout << endl;

    // int LengthOver_5 = 0;
    // ofstream save_track_distri;
    // string save_td = "track_distribution_static.txt";
    // save_track_distri.open(save_td.c_str(),ios::trunc);
    // for (int i = 0; i < N; ++i){
    //     if(TrackLength[i]!=0)
    //         save_track_distri << TrackLength[i] << endl;
    //     if (i+2>=5)
    //         LengthOver_5 = LengthOver_5 + TrackLength[i];
    // }
    // save_track_distri.close();
    //cout << "Length over 5 (STATIC):::::::::::::::: " << LengthOver_5 << endl;

    return TrackLets;
}

std::vector<std::vector<std::pair<int, int> > > Tracking::GetDynamicTrackNew()
{
    // Get temporal match from Map
    std::vector<std::vector<int> > TemporalMatch = mpMap->vnAssoDyn;
    std::vector<std::vector<int> > ObjLab = mpMap->vnFeatLabel;
    int N = TemporalMatch.size();
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;
    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;
    // save object id of each tracklets
    std::vector<int> ObjectID;

    // main loop
    int IDsofar = 0;
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(TemporalMatch[i].size(),-1);

        // check each feature
        for (int j = 0; j < TemporalMatch[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID, pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);
                    ObjectID.push_back(ObjLab[i][j]);

                    // save tracklet ID
                    TrackCheck_cur[j] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
            }
            // frame i and i+1 (i>0)
            else
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    if (TrackCheck_pre[TemporalMatch[i][j]]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j]);

                        // save tracklet ID
                        TrackCheck_cur[j] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
            }
        }

        TrackCheck_pre = TrackCheck_cur;
    }

    // update object ID list
    mpMap->nObjID = ObjectID;

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    // for (int i = 0; i < N; ++i){
    //     if(TrackLength[i]!=0)
    //         cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    // }
    // cout << endl;

    // int LengthOver_5 = 0;
    // ofstream save_track_distri;
    // string save_td = "track_distribution.txt";
    // save_track_distri.open(save_td.c_str(),ios::trunc);
    // for (int i = 0; i < N; ++i){
    //     if(TrackLength[i]!=0)
    //         save_track_distri << TrackLength[i] << endl;
    //     if (i+2>=5)
    //         LengthOver_5 = LengthOver_5 + TrackLength[i];
    // }
    // save_track_distri.close();
   // cout << "Length over 5 (DYNAMIC):::::::::::::::: " << LengthOver_5 << endl;

    return TrackLets;
}

std::vector<std::vector<int> > Tracking::GetObjTrackTime(std::vector<std::vector<int> > &ObjTrackLab, std::vector<std::vector<int> > &ObjSemanticLab,
                                                         std::vector<std::vector<int> > &vnSMLabGT)
{
    std::vector<int> TrackCount(max_id-1,0);
    std::vector<int> TrackCountGT(max_id-1,0);
    std::vector<int> SemanticLabel(max_id-1,0);
    std::vector<std::vector<int> > ObjTrackTime;

    // count each object track
    for (int i = 0; i < ObjTrackLab.size(); ++i)
    {
        if (ObjTrackLab[i].size()<2)
            continue;

        for (int j = 1; j < ObjTrackLab[i].size(); ++j)
        {
            // TrackCountGT[ObjTrackLab[i][j]-1] = TrackCountGT[ObjTrackLab[i][j]-1] + 1;
            TrackCount[ObjTrackLab[i][j]-1] = TrackCount[ObjTrackLab[i][j]-1] + 1;
            SemanticLabel[ObjTrackLab[i][j]-1] = ObjSemanticLab[i][j];
        }
    }

    // count each object track in ground truth
    for (int i = 0; i < vnSMLabGT.size(); ++i)
    {
        for (int j = 0; j < vnSMLabGT[i].size(); ++j)
        {
            for (int k = 0; k < SemanticLabel.size(); ++k)
            {
                if (SemanticLabel[k]==vnSMLabGT[i][j])
                {
                    TrackCountGT[k] = TrackCountGT[k] + 1;
                    break;
                }
            }
        }
    }

    mpMap->nObjTraCount = TrackCount;
    mpMap->nObjTraCountGT = TrackCountGT;
    mpMap->nObjTraSemLab = SemanticLabel;


    // // // show the object track count
    // cout << "Current Object Track Counting: " << endl;
    // int TotalCount = 0;
    // for (int i = 0; i < TrackCount.size(); ++i)
    // {
    //     TotalCount = TotalCount + TrackCount[i];
    //     cout << "Object " << i+1 << " has been tracked " << TrackCount[i] << " times." << endl;
    // }
    // cout << "Total Object Track Counting: " << TotalCount << endl;

    // save to each frame the count number (ObjTrackTime)
    for (int i = 0; i < ObjTrackLab.size(); ++i)
    {
        std::vector<int> TrackTimeTmp(ObjTrackLab[i].size(),0);

        if (TrackTimeTmp.size()<2)
        {
            ObjTrackTime.push_back(TrackTimeTmp);
            continue;
        }

        for (int j = 1; j < TrackTimeTmp.size(); ++j)
        {
            TrackTimeTmp[j] = TrackCount[ObjTrackLab[i][j]-1];
        }
        ObjTrackTime.push_back(TrackTimeTmp);
    }

    return ObjTrackTime;
}

std::vector<std::vector<std::pair<int, int> > > Tracking::GetDynamicTrack()
{
    std::vector<std::vector<cv::KeyPoint> > Feats = mpMap->vpFeatDyn;
    std::vector<std::vector<int> > ObjLab = mpMap->vnFeatLabel;
    int N = Feats.size();

    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;
    // save object id of each tracklets
    std::vector<int> ObjectID;
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;


    // main loop
    int IDsofar = 0;
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(Feats[i].size(),-1);

        // Check empty
        if (Feats[i].empty())
        {
           TrackCheck_pre = TrackCheck_cur;
           continue;
        }

        // first pair of frames (frame 0 and 1)
        if (i==0)
        {
            int M = Feats[i].size();
            for (int j = 0; j < M; j=j+2)
            {
                // first, save one tracklet consisting of two featureID
                std::vector<std::pair<int, int> > TraLet(2);
                TraLet[0] = std::make_pair(i,j);
                TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                // then, save to the main tracklets list
                TrackLets.push_back(TraLet);
                ObjectID.push_back(ObjLab[i][j/2]);

                // finally, save tracklet ID
                TrackCheck_cur[j+1] = IDsofar;
                IDsofar = IDsofar + 1;
            }
        }
        // frame i and i+1 (i>0)
        else
        {
            int M_pre = TrackCheck_pre.size();
            int M_cur = Feats[i].size();

            if (M_pre==0)
            {
                for (int j = 0; j < M_cur; j=j+2)
                {
                    // first, save one tracklet consisting of two featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,j);
                    TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);
                    ObjectID.push_back(ObjLab[i][j/2]);

                    // finally, save tracklet ID
                    TrackCheck_cur[j+1] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
            }
            else
            {
                // (1) find the temporal matching list (TM) between
                // previous flow locations and current sampled locations
                vector<int> TM(M_cur,-1);
                std::vector<float> MinDist(M_cur,-1);
                int nmatches = 0;
                for (int k = 1; k < M_pre; k=k+2)
                {
                    float x_ = Feats[i-1][k].pt.x;
                    float y_ = Feats[i-1][k].pt.y;
                    float min_dist = 10;
                    int candi = -1;
                    for (int j = 0; j < M_cur; j=j+2)
                    {
                        if (ObjLab[i-1][(k-1)/2]!=ObjLab[i][j/2])
                            continue;

                        float x  = Feats[i][j].pt.x;
                        float y  = Feats[i][j].pt.y;
                        float dist = std::sqrt( (x_-x)*(x_-x) + (y_-y)*(y_-y) );

                        if (dist<min_dist){
                            min_dist = dist;
                            candi = j;
                        }
                    }
                    // threshold
                    if (min_dist<1.0)
                    {
                        // current feature not occupied -or- occupied but new distance is smaller
                        // then label current match
                        if (TM[candi]==-1 || (TM[candi]!=-1 && min_dist<MinDist[candi]))
                        {
                            TM[candi] = k;
                            MinDist[candi] = min_dist;
                            nmatches = nmatches + 1;
                        }
                    }
                }

                // (2) save tracklets according to TM
                for (int j = 0; j < M_cur; j=j+2)
                {
                    // check the TM. if it is associated with last frame, then add to existing tracklets.
                    if (TM[j]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TM[j]]].push_back(std::make_pair(i,j+1)); // used to be i+1
                        TrackCheck_cur[j+1] = TrackCheck_pre[TM[j]];
                    }
                    else
                    {
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,j);
                        TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j/2]);

                        // save tracklet ID
                        TrackCheck_cur[j+1] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
            }
        }

        TrackCheck_pre = TrackCheck_cur;
    }

    // update object ID list
    mpMap->nObjID = ObjectID;


    // display info
    cout << endl;
    cout << "==============================================" << endl;
    cout << "the number of object feature tracklets: " << TrackLets.size() << endl;
    cout << "==============================================" << endl;
    cout << endl;

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    for (int i = 0; i < N; ++i)
        cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    cout << endl;


    return TrackLets;
}

void Tracking::RenewFrameInfo(const std::vector<int> &TM_sta)
{
    Verbose::PrintMess("Start Renew Frame Information",Verbose::VERBOSITY_DEBUG);
    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ Update for static features +++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    // use sampled or detected features
    int max_num_sta = nMaxTrackPointBG;
    int max_num_obj = nMaxTrackPointOBJ;

    std::vector<cv::KeyPoint> mvKeysTmp;
    std::vector<cv::KeyPoint> mvCorresTmp;
    std::vector<cv::Point2f> mvFlowNextTmp;
    std::vector<int> StaInlierIDTmp;

    // (1) Save the inliers from last frame
    for (int i = 0; i < TM_sta.size(); ++i)
    {
        if (TM_sta[i]==-1)
            continue;

        int x = mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.x;
        int y = mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.y;

        if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
            continue;

        if (mSegMap.at<int>(y,x)!=0)
            continue;

        if (mDepthMap.at<float>(y,x)>40 || mDepthMap.at<float>(y,x)<=0)
            continue;

        float flow_xe = mFlowMap.at<cv::Vec2f>(y,x)[0];
        float flow_ye = mFlowMap.at<cv::Vec2f>(y,x)[1];

        if(flow_xe!=0 && flow_ye!=0)
        {
            if(mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.x+flow_xe < mImGrayLast.cols && mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.y+flow_ye < mImGrayLast.rows && mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.x+flow_xe>0 && mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.y+flow_ye>0)
            {
                mvKeysTmp.push_back(mpCurrentFrame->mvStatKeys[TM_sta[i]]);
                mvCorresTmp.push_back(cv::KeyPoint(mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.x+flow_xe,mpCurrentFrame->mvStatKeys[TM_sta[i]].pt.y+flow_ye,0,0,0,-1));
                mvFlowNextTmp.push_back(cv::Point2f(flow_xe,flow_ye));
                StaInlierIDTmp.push_back(TM_sta[i]);
            }
        }

        if (mvKeysTmp.size()>max_num_sta)
            break;
    }

    // (2) Save extra keypoints to make it a fixed number 
    int tot_num = mvKeysTmp.size(), start_id = 0, step = 20;
    std::vector<cv::KeyPoint> mvKeysTmpCheck = mvKeysTmp;
    std::vector<cv::KeyPoint> mvKeysSample;
    if (nUseSampleFea==1)
        mvKeysSample = mpCurrentFrame->mvStatKeysTmp;
    else
        mvKeysSample = mpCurrentFrame->mvKeys;
    while (tot_num<max_num_sta)
    {
        // start id > step number, then stop
        if (start_id==step)
            break;

        for (int i = start_id; i < mvKeysSample.size(); i=i+step)
        {
            // check if this key point is already been used
            float min_dist = 100;
            bool used = false;
            for (int j = 0; j < mvKeysTmpCheck.size(); ++j)
            {
                float cur_dist = std::sqrt( (mvKeysTmpCheck[j].pt.x-mvKeysSample[i].pt.x)*(mvKeysTmpCheck[j].pt.x-mvKeysSample[i].pt.x) + (mvKeysTmpCheck[j].pt.y-mvKeysSample[i].pt.y)*(mvKeysTmpCheck[j].pt.y-mvKeysSample[i].pt.y) );
                if (cur_dist<min_dist)
                    min_dist = cur_dist;
                if (min_dist<1.0)
                {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            int x = mvKeysSample[i].pt.x;
            int y = mvKeysSample[i].pt.y;

            if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
                continue;

            if (mSegMap.at<int>(y,x)!=0)
                continue;

            if (mDepthMap.at<float>(y,x)>40 || mDepthMap.at<float>(y,x)<=0)
                continue;

            float flow_xe = mFlowMap.at<cv::Vec2f>(y,x)[0];
            float flow_ye = mFlowMap.at<cv::Vec2f>(y,x)[1];

            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeysSample[i].pt.x+flow_xe < mImGrayLast.cols && mvKeysSample[i].pt.y+flow_ye < mImGrayLast.rows && mvKeysSample[i].pt.x+flow_xe > 0 && mvKeysSample[i].pt.y+flow_ye > 0)
                {
                    mvKeysTmp.push_back(mvKeysSample[i]);
                    mvCorresTmp.push_back(cv::KeyPoint(mvKeysSample[i].pt.x+flow_xe,mvKeysSample[i].pt.y+flow_ye,0,0,0,-1));
                    mvFlowNextTmp.push_back(cv::Point2f(flow_xe,flow_ye));
                    StaInlierIDTmp.push_back(-1);
                    tot_num = tot_num + 1;
                }
            }

            if (tot_num>=max_num_sta)
                break;
        }
        start_id = start_id + 1;
    }

    mpCurrentFrame->N_s_tmp = mvKeysTmp.size();

    // (3) assign the depth value to each key point
    std::vector<float> mvDepthTmp(mpCurrentFrame->N_s_tmp,-1);
    for(int i=0; i<mpCurrentFrame->N_s_tmp; i++)
    {
        const cv::KeyPoint &kp = mvKeysTmp[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        float d = mDepthMap.at<float>(v,u); // be careful with the order  !!!

        if(d>0)
            mvDepthTmp[i] = d;
    }

    // (4) create 3d point based on key point, depth and pose
    std::vector<cv::Mat> mv3DPointTmp(mpCurrentFrame->N_s_tmp);
    for (int i = 0; i < mpCurrentFrame->N_s_tmp; ++i)
    {
        mv3DPointTmp[i] = Optimizer::Get3DinWorld(mvKeysTmp[i], mvDepthTmp[i], mK, Converter::toInvMatrix(mpCurrentFrame->mTcw));
    }

    // Obtain inlier ID
    mpCurrentFrame->nStaInlierID = StaInlierIDTmp;

    // Update
    mpCurrentFrame->mvStatKeysTmp = mvKeysTmp;
    mpCurrentFrame->mvStatDepthTmp = mvDepthTmp;
    mpCurrentFrame->mvStat3DPointTmp = mv3DPointTmp;
    mpCurrentFrame->mvFlowNext = mvFlowNextTmp;
    mpCurrentFrame->mvCorres = mvCorresTmp;

    // cout << "updating STATIC features finished...... " << mvKeysTmp.size() << endl;

    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ Update for Dynamic Object Features +++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    std::vector<cv::KeyPoint> mvObjKeysTmp;
    std::vector<float> mvObjDepthTmp;
    std::vector<cv::KeyPoint> mvObjCorresTmp;
    std::vector<cv::Point2f> mvObjFlowNextTmp;
    std::vector<int> vSemObjLabelTmp;
    std::vector<int> DynInlierIDTmp;
    std::vector<int> vObjLabelTmp;

    // (1) Again, save the inliers from last frame
    std::vector<std::vector<int> > ObjInlierSet = mpCurrentFrame->vnObjInlierID;
    std::vector<int> ObjFeaCount(ObjInlierSet.size());
    for (int i = 0; i < ObjInlierSet.size(); ++i)
    {
        // remove failure object
        if (!mpCurrentFrame->bObjStat[i])
        {
            ObjFeaCount[i] = -1;
            continue;
        }

        int count = 0;
        for (int j = 0; j < ObjInlierSet[i].size(); ++j)
        {
            const int x = mpCurrentFrame->mvObjKeys[ObjInlierSet[i][j]].pt.x;
            const int y = mpCurrentFrame->mvObjKeys[ObjInlierSet[i][j]].pt.y;

            if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
                continue;

            if (mSegMap.at<int>(y,x)!=0 && mDepthMap.at<float>(y,x)<25 && mDepthMap.at<float>(y,x)>0)
            {
                const float flow_x = mFlowMap.at<cv::Vec2f>(y,x)[0];
                const float flow_y = mFlowMap.at<cv::Vec2f>(y,x)[1];

                if (x+flow_x < mImGrayLast.cols && y+flow_y < mImGrayLast.rows && x+flow_x>0 && y+flow_y>0)
                {
                    mvObjKeysTmp.push_back(cv::KeyPoint(x,y,0,0,0,-1));
                    mvObjDepthTmp.push_back(mDepthMap.at<float>(y,x));
                    vSemObjLabelTmp.push_back(mSegMap.at<int>(y,x));
                    mvObjFlowNextTmp.push_back(cv::Point2f(flow_x,flow_y));
                    mvObjCorresTmp.push_back(cv::KeyPoint(x+flow_x,y+flow_y,0,0,0,-1));
                    DynInlierIDTmp.push_back(ObjInlierSet[i][j]);
                    vObjLabelTmp.push_back(mpCurrentFrame->vObjLabel[ObjInlierSet[i][j]]);
                    count = count + 1;
                }
            }
        }
        ObjFeaCount[i] = count;
        // cout << "accumulate dynamic inlier number: " << ObjFeaCount[i] << endl;
    }


    // (2) Save extra key points to make each object having a fixed number (max = 400, 800, 1000)
    std::vector<std::vector<int> > ObjSet = mpCurrentFrame->vnObjID;
    std::vector<cv::KeyPoint> mvObjKeysTmpCheck = mvObjKeysTmp;
    for (int i = 0; i < ObjSet.size(); ++i)
    {
        // remove failure object
        if (!mpCurrentFrame->bObjStat[i])
            continue;

        int SemLabel = mpCurrentFrame->nSemPosition[i];
        int tot_num = ObjFeaCount[i];
        int start_id = 0, step = 15;
        while (tot_num<max_num_obj)
        {
            // start id > step number, then stop
            if (start_id==step){
                // cout << "run on all the original objset... tot_num: " << tot_num << endl;
                break;
            }

            for (int j = start_id; j < mvTmpSemObjLabel.size(); j=j+step)
            {
                // check the semantic label if it is the same
                if (mvTmpSemObjLabel[j]!=SemLabel)
                    continue;

                // check if this key point is already been used
                float min_dist = 100;
                bool used = false;
                for (int k = 0; k < mvObjKeysTmpCheck.size(); ++k)
                {
                    float cur_dist = std::sqrt( (mvObjKeysTmpCheck[k].pt.x-mvTmpObjKeys[j].pt.x)*(mvObjKeysTmpCheck[k].pt.x-mvTmpObjKeys[j].pt.x) + (mvObjKeysTmpCheck[k].pt.y-mvTmpObjKeys[j].pt.y)*(mvObjKeysTmpCheck[k].pt.y-mvTmpObjKeys[j].pt.y) );
                    if (cur_dist<min_dist)
                        min_dist = cur_dist;
                    if (min_dist<1.0)
                    {
                        used = true;
                        break;
                    }
                }
                if (used)
                    continue;

                // save the found one
                mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                mvObjDepthTmp.push_back(mvTmpObjDepth[j]);
                vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                DynInlierIDTmp.push_back(-1);
                vObjLabelTmp.push_back(mpCurrentFrame->nModLabel[i]);
                tot_num = tot_num + 1;

                if (tot_num>=max_num_obj){
                    // cout << "reach max_num_obj... tot_num: " << tot_num << endl;
                    break;
                }
            }
            start_id = start_id + 1;
        }

    }

    // (3) Update new appearing objects
    // (3.1) find the unique labels in semantic label
    auto UniLab = mvTmpSemObjLabel;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );
    // (3.2) find new appearing label
    std::vector<bool> NewLab(UniLab.size(),false);
    for (int i = 0; i < mpCurrentFrame->nSemPosition.size(); ++i)
    {
        int CurSemLabel = mpCurrentFrame->nSemPosition[i];
        for (int j = 0; j < UniLab.size(); ++j)
        {
            if (UniLab[j]==CurSemLabel && mpCurrentFrame->bObjStat[i]) // && mpCurrentFrame->bObjStat[i]
            {
                NewLab[j] = true;
                break;
            }
        }

    }
    // (3.3) add the new object key points
    for (int i = 0; i < NewLab.size(); ++i)
    {
        if (NewLab[i]==false)
        {
            for (int j = 0; j < mvTmpSemObjLabel.size(); j++)
            {
                if (UniLab[i]==mvTmpSemObjLabel[j])
                {
                    // save the found one
                    mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                    mvObjDepthTmp.push_back(mvTmpObjDepth[j]);
                    vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                    mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                    mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                    DynInlierIDTmp.push_back(-1);
                    vObjLabelTmp.push_back(-2);
                }
            }
        }
    }

    // (4) create 3d point based on key point, depth and pose
    std::vector<cv::Mat> mvObj3DPointTmp(mvObjKeysTmp.size());
    for (int i = 0; i < mvObjKeysTmp.size(); ++i)
        mvObj3DPointTmp[i] = Optimizer::Get3DinWorld(mvObjKeysTmp[i], mvObjDepthTmp[i], mK, Converter::toInvMatrix(mpCurrentFrame->mTcw));


    // update
    mpCurrentFrame->mvObjKeys = mvObjKeysTmp;
    mpCurrentFrame->mvObjDepth = mvObjDepthTmp;
    mpCurrentFrame->mvObj3DPoint = mvObj3DPointTmp;
    mpCurrentFrame->mvObjCorres = mvObjCorresTmp;
    mpCurrentFrame->mvObjFlowNext = mvObjFlowNextTmp;
    mpCurrentFrame->vSemObjLabel = vSemObjLabelTmp;
    mpCurrentFrame->nDynInlierID = DynInlierIDTmp;
    mpCurrentFrame->vObjLabel = vObjLabelTmp;
}

void Tracking::UpdateMask()
{

    // find the unique labels in semantic label
    auto UniLab = mpLastFrame->vSemObjLabel;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );
    // collect the predicted labels and semantic labels in vector
    std::vector<std::vector<int> > ObjID(UniLab.size());
    for (int i = 0; i < mpLastFrame->vSemObjLabel.size(); ++i)
    {
        // save object label
        for (int j = 0; j < UniLab.size(); ++j)
        {
            if(mpLastFrame->vSemObjLabel[i]==UniLab[j]){
                ObjID[j].push_back(i);
                break;
            }
        }
    }
    // check each object label distribution in the coming frame
    for (int i = 0; i < ObjID.size(); ++i)
    {
        // collect labels
        std::vector<int> LabTmp;
        for (int j = 0; j < ObjID[i].size(); ++j)
        {
            const int u = mpLastFrame->mvObjCorres[ObjID[i][j]].pt.x;
            const int v = mpLastFrame->mvObjCorres[ObjID[i][j]].pt.y;
            if (u<mImGray.cols && u>0 && v<mImGray.rows && v>0)
            {
                LabTmp.push_back(mSegMap.at<int>(v,u));
            }
        }
        if (LabTmp.size()<100)
            continue;

        // find label that appears most in LabTmp()
        // (1) count duplicates
        std::map<int, int> dups;
        for(int k : LabTmp)
            ++dups[k];
        // (2) and sort them by descending order
        std::vector<std::pair<int, int> > sorted;
        for (auto k : dups)
            sorted.push_back(std::make_pair(k.first,k.second));
        std::sort(sorted.begin(), sorted.end(), SortPairInt);
        // recover the missing mask (time consuming!)
        if (sorted[0].first==0) // no mask 
        {
            for (int j = 0; j < mImGrayLast.rows; j++)
            {
                for (int k = 0; k < mImGrayLast.cols; k++)
                {
                    if (mSegMapLast.at<int>(j,k)==UniLab[i]) //如果上一帧分割图的像素标签是当前标签
                    {
                        const int flow_x = mFlowMapLast.at<cv::Vec2f>(j,k)[0];
                        const int flow_y = mFlowMapLast.at<cv::Vec2f>(j,k)[1];

                        if(k+flow_x < mImGrayLast.cols && k+flow_x > 0 && j+flow_y < mImGrayLast.rows && j+flow_y > 0)
                            mSegMap.at<int>(j+flow_y,k+flow_x) = UniLab[i];//给本帧分割图(j+fow_y,k+flow_x)像素添加标签属性
                    }
                }
            }
        }
        // end of recovery
    }
   
    // // === verify the updated labels ===
    // cv::Mat imgLabel(mImGray.rows,mImGray.cols,CV_8UC3); // for display
    // for (int i = 0; i < mSegMap.rows; ++i)
    // {
    //     for (int j = 0; j < mSegMap.cols; ++j)
    //     {
    //         int tmp = mSegMap.at<int>(i,j);
    //         if (tmp>50)
    //             tmp = tmp/2;
    //         switch (tmp)
    //         {
    //             case 0:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,240);
    //                 break;
    //             case 1:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);
    //                 break;
    //             case 2:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);
    //                 break;
    //             case 3:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,0);
    //                 break;
    //             case 4:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(47,255,173); // greenyellow
    //                 break;
    //             case 5:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 128);
    //                 break;
    //             case 6:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(203,192,255);
    //                 break;
    //             case 7:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(196,228,255);
    //                 break;
    //             case 8:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(42,42,165);
    //                 break;
    //             case 9:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
    //                 break;
    //             case 10:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(245,245,245); // whitesmoke
    //                 break;
    //             case 11:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,165,255); // orange
    //                 break;
    //             case 12:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(230,216,173); // lightblue
    //                 break;
    //             case 13:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128,128,128); // grey
    //                 break;
    //             case 14:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,215,255); // gold
    //                 break;
    //             case 15:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
    //                 break;
    //             case 16:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
    //                 break;
    //             case 17:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 18:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 19:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 20:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //             case 21:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
    //                 break;
    //             case 22:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
    //                 break;
    //             case 23:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
    //                 break;
    //             case 24:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
    //                 break;
    //             case 25:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 255, 127); // chartreuse
    //                 break;
    //             case 26:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(139, 0, 0);  // darkblue
    //                 break;
    //             case 27:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(60, 20, 220);  // crimson
    //                 break;
    //             case 28:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 139);  // darkred
    //                 break;
    //             case 29:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(211, 0, 148);  // darkviolet
    //                 break;
    //             case 30:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 144, 30);  // dodgerblue
    //                 break;
    //             case 31:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(105, 105, 105);  // dimgray
    //                 break;
    //             case 32:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(180, 105, 255);  // hotpink
    //                 break;
    //             case 33:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(204, 209, 72);  // mediumturquoise
    //                 break;
    //             case 34:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(173, 222, 255);  // navajowhite
    //                 break;
    //             case 35:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(143, 143, 188); // rosybrown
    //                 break;
    //             case 36:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(50, 205, 50);  // limegreen
    //                 break;
    //             case 37:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 38:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 39:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 40:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //             case 41:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
    //                 break;
    //             case 42:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
    //                 break;
    //             case 43:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
    //                 break;
    //             case 44:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
    //                 break;
    //             case 45:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
    //                 break;
    //             case 46:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
    //                 break;
    //             case 47:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 48:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 49:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 50:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //         }
    //     }
    // }
    // cv::imshow("Updated Mask Image", imgLabel);
    // cv::waitKey(1);
    //cout << "Update Mask, Done!" << endl;
}

void Tracking::GetMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                    const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                    const std::vector<std::vector<bool> > &ObjStat)
{
    bool bRMSError = false;
    cout << "=================================================" << endl;

    // absolute trajectory error for CAMERA (RMSE)
    cout << "CAMERA:" << endl;
    float t_sum = 0, r_sum = 0;
    for (int i = 1; i < CamPose.size(); ++i)
    {
        cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
        cv::Mat T_lc_gt = CamPose_gt[i-1]*Converter::toInvMatrix(CamPose_gt[i]);
        cv::Mat ate_cam = T_lc_inv*T_lc_gt;
        // cv::Mat ate_cam = CamPose[i]*Converter::toInvMatrix(CamPose_gt[i]);

        // translation
        float t_ate_cam = std::sqrt(ate_cam.at<float>(0,3)*ate_cam.at<float>(0,3) + ate_cam.at<float>(1,3)*ate_cam.at<float>(1,3) + ate_cam.at<float>(2,3)*ate_cam.at<float>(2,3));
        if (bRMSError)
            t_sum = t_sum + t_ate_cam*t_ate_cam;
        else
            t_sum = t_sum + t_ate_cam;

        // rotation
        float trace_ate = 0;
        for (int j = 0; j < 3; ++j)
        {
            if (ate_cam.at<float>(j,j)>1.0)
                trace_ate = trace_ate + 1.0-(ate_cam.at<float>(j,j)-1.0);
            else
                trace_ate = trace_ate + ate_cam.at<float>(j,j);
        }
        float r_ate_cam = acos( (trace_ate -1.0)/2.0 )*180.0/3.1415926;
        if (bRMSError)
            r_sum = r_sum + r_ate_cam*r_ate_cam;
        else
            r_sum = r_sum + r_ate_cam;

        // cout << " t: " << t_ate_cam << " R: " << r_ate_cam << endl;
    }
    if (bRMSError)
    {
        t_sum = std::sqrt(t_sum/(CamPose.size()-1));
        r_sum = std::sqrt(r_sum/(CamPose.size()-1));
    }
    else
    {
        t_sum = t_sum/(CamPose.size()-1);
        r_sum = r_sum/(CamPose.size()-1);
    }

    cout << "average error (Camera):" << " t: " << t_sum << " R: " << r_sum << endl;

    std::vector<float> each_obj_t(max_id-1,0);
    std::vector<float> each_obj_r(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    // all motion error for OBJECTS (mean error)
    cout << "OBJECTS:" << endl;
    float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
    for (int i = 0; i < RigMot.size(); ++i)
    {
        if (RigMot[i].size()>1)
        {
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody)*RigMot_gt[i][j];

                // translation error
                float t_rpe_obj = std::sqrt( rpe_obj.at<float>(0,3)*rpe_obj.at<float>(0,3) + rpe_obj.at<float>(1,3)*rpe_obj.at<float>(1,3) + rpe_obj.at<float>(2,3)*rpe_obj.at<float>(2,3) );
                if (bRMSError){
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj*t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj*t_rpe_obj;
                }
                else{
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj;
                }

                // rotation error
                float trace_rpe = 0;
                for (int k = 0; k < 3; ++k)
                {
                    if (rpe_obj.at<float>(k,k)>1.0)
                        trace_rpe = trace_rpe + 1.0-(rpe_obj.at<float>(k,k)-1.0);
                    else
                        trace_rpe = trace_rpe + rpe_obj.at<float>(k,k);
                }
                float r_rpe_obj = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926;
                if (bRMSError){
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj*r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj*r_rpe_obj;
                }
                else{
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj;
                }

                // cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " t: " << t_rpe_obj << " R: " << r_rpe_obj << endl;
                obj_count++;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
            }
        }
    }
    if (bRMSError)
    {
        t_rpe_sum = std::sqrt(t_rpe_sum/obj_count);
        r_rpe_sum = std::sqrt(r_rpe_sum/obj_count);
    }
    else
    {
        t_rpe_sum = t_rpe_sum/obj_count;
        r_rpe_sum = r_rpe_sum/obj_count;
    }
    cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;

    // show each object
    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError)
        {
            each_obj_t[i] = std::sqrt(each_obj_t[i]/each_obj_count[i]);
            each_obj_r[i] = std::sqrt(each_obj_r[i]/each_obj_count[i]);
        }
        else
        {
            each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
            each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
        }
        if (each_obj_count[i]>=3)
            cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << endl;
    }

    cout << "=================================================" << endl;

}

void Tracking::PlotMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                    const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                    const std::vector<std::vector<bool> > &ObjStat)
{
    // saved evaluated errors
    std::vector<float> CamRotErr(CamPose.size()-1);
    std::vector<float> CamTraErr(CamPose.size()-1);
    std::vector<std::vector<float> > ObjRotErr(max_id-1);
    std::vector<std::vector<float> > ObjTraErr(max_id-1);

    bool bRMSError = false, bAccumError = true;
    cout << "=================================================" << endl;

    // absolute trajectory error for CAMERA (RMSE)
    cout << "CAMERA:" << endl;
    float t_sum = 0, r_sum = 0;
    for (int i = 1; i < CamPose.size(); ++i)
    {
        cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
        cv::Mat T_lc_gt = CamPose_gt[i-1]*Converter::toInvMatrix(CamPose_gt[i]);
        cv::Mat ate_cam = T_lc_inv*T_lc_gt;
        // cv::Mat ate_cam = CamPose[i]*Converter::toInvMatrix(CamPose_gt[i]);

        // translation
        float t_ate_cam = std::sqrt(ate_cam.at<float>(0,3)*ate_cam.at<float>(0,3) + ate_cam.at<float>(1,3)*ate_cam.at<float>(1,3) + ate_cam.at<float>(2,3)*ate_cam.at<float>(2,3));
        if (bRMSError)
            t_sum = t_sum + t_ate_cam*t_ate_cam;
        else
            t_sum = t_sum + t_ate_cam;

        // rotation
        float trace_ate = 0;
        for (int j = 0; j < 3; ++j)
        {
            if (ate_cam.at<float>(j,j)>1.0)
                trace_ate = trace_ate + 1.0-(ate_cam.at<float>(j,j)-1.0);
            else
                trace_ate = trace_ate + ate_cam.at<float>(j,j);
        }
        float r_ate_cam = acos( (trace_ate -1.0)/2.0 )*180.0/3.1415926;
        if (bRMSError)
            r_sum = r_sum + r_ate_cam*r_ate_cam;
        else
            r_sum = r_sum + r_ate_cam;

        if (bAccumError)
        {
            CamRotErr[i-1] = r_ate_cam/i;
            CamTraErr[i-1] = t_ate_cam/i;
        }
        else
        {
            CamRotErr[i-1] = r_ate_cam;
            CamTraErr[i-1] = t_ate_cam;            
        }




        // cout << " t: " << t_ate_cam << " R: " << r_ate_cam << endl;
    }
    if (bRMSError)
    {
        t_sum = std::sqrt(t_sum/(CamPose.size()-1));
        r_sum = std::sqrt(r_sum/(CamPose.size()-1));
    }
    else
    {
        t_sum = t_sum/(CamPose.size()-1);
        r_sum = r_sum/(CamPose.size()-1);
    }

    cout << "average error (Camera):" << " t: " << t_sum << " R: " << r_sum << endl;

    std::vector<float> each_obj_t(max_id-1,0);
    std::vector<float> each_obj_r(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    // all motion error for OBJECTS (mean error)
    cout << "OBJECTS:" << endl;
    float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
    for (int i = 0; i < RigMot.size(); ++i)
    {
        if (RigMot[i].size()>1)
        {
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody)*RigMot_gt[i][j];

                // translation error
                float t_rpe_obj = std::sqrt( rpe_obj.at<float>(0,3)*rpe_obj.at<float>(0,3) + rpe_obj.at<float>(1,3)*rpe_obj.at<float>(1,3) + rpe_obj.at<float>(2,3)*rpe_obj.at<float>(2,3) );
                if (bRMSError){
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj*t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj*t_rpe_obj;
                }
                else{
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj;
                }

                // rotation error
                float trace_rpe = 0;
                for (int k = 0; k < 3; ++k)
                {
                    if (rpe_obj.at<float>(k,k)>1.0)
                        trace_rpe = trace_rpe + 1.0-(rpe_obj.at<float>(k,k)-1.0);
                    else
                        trace_rpe = trace_rpe + rpe_obj.at<float>(k,k);
                }
                float r_rpe_obj = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926; 
                if (bRMSError){
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj*r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj*r_rpe_obj;
                }
                else{
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj;
                }

                // cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " t: " << t_rpe_obj << " R: " << r_rpe_obj << endl;
                obj_count++;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
                if (bAccumError)
                {
                    ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_t[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                    ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_r[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                }
                else
                {
                    ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(t_rpe_obj);
                    ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(r_rpe_obj);           
                }

            }
        }
    }
    if (bRMSError)
    {
        t_rpe_sum = std::sqrt(t_rpe_sum/obj_count);
        r_rpe_sum = std::sqrt(r_rpe_sum/obj_count);
    }
    else
    {
        t_rpe_sum = t_rpe_sum/obj_count;
        r_rpe_sum = r_rpe_sum/obj_count;
    }
    cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;

    // show each object
    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError)
        {
            each_obj_t[i] = std::sqrt(each_obj_t[i]/each_obj_count[i]);
            each_obj_r[i] = std::sqrt(each_obj_r[i]/each_obj_count[i]);
        }
        else
        {
            each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
            each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
        }
        if (each_obj_count[i]>=3)
            cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << endl;
    }

    cout << "=================================================" << endl;


    auto name1 = "Translation";
    cvplot::setWindowTitle(name1, "Translation Error (Meter)");
    cvplot::moveWindow(name1, 0, 240);
    cvplot::resizeWindow(name1, 800, 240);
    auto &figure1 = cvplot::figure(name1);

    auto name2 = "Rotation";
    cvplot::setWindowTitle(name2, "Rotation Error (Degree)");
    cvplot::resizeWindow(name2, 800, 240);
    auto &figure2 = cvplot::figure(name2);

    figure1.series("Camera")
        .setValue(CamTraErr)
        .type(cvplot::DotLine)
        .color(cvplot::Red);

    figure2.series("Camera")
        .setValue(CamRotErr)
        .type(cvplot::DotLine)
        .color(cvplot::Red);

    for (int i = 0; i < max_id-1; ++i)
    {
        switch (i)
        {
            case 0:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Purple);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Purple);
                break;
            case 1:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Green);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Green);
                break;
            case 2:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Cyan);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Cyan);
                break;
            case 3:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Blue);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Blue);
                break;
            case 4:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Pink);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Pink);
                break;
        }
    }

    figure1.show(true);
    figure2.show(true);

}

void Tracking::GetVelocityError(const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &PointDyn,
                                const std::vector<std::vector<int> > &FeaLab, const std::vector<std::vector<int> > &RMLab,
                                const std::vector<std::vector<float> > &Velo_gt, const std::vector<std::vector<int> > &TmpMatch,
                                const std::vector<std::vector<bool> > &ObjStat)
{
    bool bRMSError = true;
    float s_sum = 0, s_gt_sum = 0, obj_count = 0;

    string path = "/Users/steed/work/code/Evaluation/ijrr2020/";
    string path_sp_e = path + "speed_error.txt";
    string path_sp_est = path + "speed_estimated.txt";
    string path_sp_gt = path + "speed_groundtruth.txt";
    string path_track = path + "tracking_id.txt";
    ofstream save_sp_e, save_sp_est, save_sp_gt, save_tra;
    save_sp_e.open(path_sp_e.c_str(),ios::trunc);
    save_sp_est.open(path_sp_est.c_str(),ios::trunc);
    save_sp_gt.open(path_sp_gt.c_str(),ios::trunc);
    save_tra.open(path_track.c_str(),ios::trunc);

    std::vector<float> each_obj_est(max_id-1,0);
    std::vector<float> each_obj_gt(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    cout << "OBJECTS SPEED:" << endl;

    // Main loop for each frame
    for (int i = 0; i < RigMot.size(); ++i)
    {
        save_tra << i << " " << 0 << " ";

        // Check if there are moving objects, and if all the variables are consistent
        if (RigMot[i].size()>1 && Velo_gt[i].size()>1 && RMLab[i].size()>1)
        {
            // Loop for each object in each frame
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                // check if this is valid object estimate
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                // (1) Compute each object centroid
                cv::Mat ObjCenter = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
                float ObjFeaCount = 0;
                if (i==0)
                {
                    for (int k = 0; k < PointDyn[i+1].size(); ++k)
                    {
                        if (FeaLab[i][k]!=RMLab[i][j])
                            continue;
                        if (TmpMatch[i][k]==-1)
                            continue;

                        ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                        ObjFeaCount = ObjFeaCount + 1;
                    }
                    ObjCenter = ObjCenter/ObjFeaCount;
                }
                else
                {
                    for (int k = 0; k < PointDyn[i+1].size(); ++k)
                    {
                        if (FeaLab[i][k]!=RMLab[i][j])
                            continue;
                        if (TmpMatch[i][k]==-1)
                            continue;

                        ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                        ObjFeaCount = ObjFeaCount + 1;
                    }
                    ObjCenter = ObjCenter/ObjFeaCount;
                }


                // (2) Compute object velocity
                cv::Mat sp_est_v = RigMot[i][j].rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-RigMot[i][j].rowRange(0,3).colRange(0,3))*ObjCenter;
                float sp_est_norm = std::sqrt( sp_est_v.at<float>(0)*sp_est_v.at<float>(0) + sp_est_v.at<float>(1)*sp_est_v.at<float>(1) + sp_est_v.at<float>(2)*sp_est_v.at<float>(2) )*36;

                // (3) Compute velocity error
                float speed_error = sp_est_norm - Velo_gt[i][j];
                if (bRMSError){
                    each_obj_est[mpMap->vnRMLabel[i][j]-1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm*sp_est_norm;
                    each_obj_gt[mpMap->vnRMLabel[i][j]-1] = each_obj_gt[mpMap->vnRMLabel[i][j]-1] + Velo_gt[i][j]*Velo_gt[i][j];
                    s_sum = s_sum + speed_error*speed_error;
                }
                else{
                    each_obj_est[mpMap->vnRMLabel[i][j]-1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm;
                    each_obj_gt[mpMap->vnRMLabel[i][j]-1] = each_obj_gt[mpMap->vnRMLabel[i][j]-1] + Velo_gt[i][j];
                    s_sum = s_sum + speed_error;
                }

                // (4) sum ground truth speed
                s_gt_sum = s_gt_sum + Velo_gt[i][j];

                save_sp_e << fixed << setprecision(6) << speed_error << endl;
                save_sp_est << fixed << setprecision(6) << sp_est_norm << endl;
                save_sp_gt << fixed << setprecision(6) << Velo_gt[i][j] << endl;
                save_tra << mpMap->vnRMLabel[i][j] << " ";

                // cout << "(" << i+1 << "/" << mpMap->vnRMLabel[i][j] << ")" << " s: " << speed_error << " est: " << sp_est_norm << " gt: " << Velo_gt[i][j] << endl;
                obj_count = obj_count + 1;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
            }
            save_tra << endl;
        }
    }

    save_sp_e.close();
    save_sp_est.close();
    save_sp_gt.close();

    if (bRMSError)
        s_sum = std::sqrt(s_sum/obj_count);
    else
        s_sum = std::abs(s_sum/obj_count);

    s_gt_sum = s_gt_sum/obj_count;

    cout << "average speed error (All Objects):" << " s: " << s_sum << "km/h " << "Track Num: " << (int)obj_count << " GT AVG SPEED: " << s_gt_sum << endl;

    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError){
            each_obj_est[i] = std::sqrt(each_obj_est[i]/each_obj_count[i]);
            each_obj_gt[i] = std::sqrt(each_obj_gt[i]/each_obj_count[i]);
        }
        else{
            each_obj_est[i] = each_obj_est[i]/each_obj_count[i];
            each_obj_gt[i] = each_obj_gt[i]/each_obj_count[i];
        }
        if (mpMap->nObjTraCount[i]>=3)
            cout << endl << "average error of Object " << i+1 << " (" << mpMap->nObjTraCount[i] << "/" << mpMap->nObjTraCountGT[i] <<  "/" << mpMap->nObjTraSemLab[i]  << "): " << " (est) " << each_obj_est[i] << " (gt) " << each_obj_gt[i] << endl;
    }

    cout << "=================================================" << endl << endl;

}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

} //namespace VDO_SLAM
