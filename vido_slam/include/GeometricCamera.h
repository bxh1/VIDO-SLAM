#ifndef CAMERAMODELS_GEOMETRICCAMERA_H
#define CAMERAMODELS_GEOMETRICCAMERA_H

#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include <Eigen/Geometry>

namespace VIDO_SLAM
 {
    class GeometricCamera {

        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & mnId;
            ar & mnType;
            ar & mvParameters;
        }


    public:
        GeometricCamera() {}
        GeometricCamera(const std::vector<float> &_vParameters) : mvParameters(_vParameters) {}
        ~GeometricCamera() {}

        virtual cv::Point2f project(const cv::Point3f &p3D) = 0;
        virtual cv::Point2f project(const cv::Mat& m3D) = 0;
        virtual Eigen::Vector2d project(const Eigen::Vector3d & v3D) = 0;
        virtual cv::Mat projectMat(const cv::Point3f& p3D) = 0;

        virtual float uncertainty2(const Eigen::Matrix<double,2,1> &p2D) = 0;

        virtual cv::Point3f unproject(const cv::Point2f &p2D) = 0;
        virtual cv::Mat unprojectMat(const cv::Point2f &p2D) = 0;

        virtual cv::Mat projectJac(const cv::Point3f &p3D) = 0;
        virtual Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D) = 0;

        virtual cv::Mat unprojectJac(const cv::Point2f &p2D) = 0;

        virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                                             cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) = 0;

        virtual cv::Mat toK() = 0;

        virtual bool epipolarConstrain(GeometricCamera* otherCamera, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc) = 0;

        float getParameter(const int i){return mvParameters[i];}
        void setParameter(const float p, const size_t i){mvParameters[i] = p;}

        size_t size(){return mvParameters.size();}

        virtual bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, GeometricCamera* pOther,
                                 cv::Mat& Tcw1, cv::Mat& Tcw2,
                                 const float sigmaLevel1, const float sigmaLevel2,
                                 cv::Mat& x3Dtriangulated) = 0;

        unsigned int GetId() { return mnId; }

        unsigned int GetType() { return mnType; }

        const unsigned int CAM_PINHOLE = 0;
        const unsigned int CAM_FISHEYE = 1;

        static long unsigned int nNextId;

    protected:
        std::vector<float> mvParameters;

        unsigned int mnId;

        unsigned int mnType;
    };
}


#endif //CAMERAMODELS_GEOMETRICCAMERA_H
