/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "G2oTypes.h"
#include "ImuTypes.h"
#include "Converter.h"
namespace VIDO_SLAM
{

ImuCamPose::ImuCamPose(Frame *pF):its(0)
{
    // Load IMU pose
    twb = Converter::toVector3d(pF->GetImuPosition());
    Rwb = Converter::toMatrix3d(pF->GetImuRotation());

    // Load camera poses
    int num_cams = 1;

    tcw.resize(num_cams);
    Rcw.resize(num_cams);
    tcb.resize(num_cams);
    Rcb.resize(num_cams);
    Rbc.resize(num_cams);
    tbc.resize(num_cams);

    // Left camera
    tcw[0] = Converter::toVector3d(pF->mTcw.rowRange(0,3).col(3));
    Rcw[0] = Converter::toMatrix3d(pF->mTcw.rowRange(0,3).colRange(0,3));
    tcb[0] = Converter::toVector3d(pF->mImuCalib.Tcb.rowRange(0,3).col(3));
    Rcb[0] = Converter::toMatrix3d(pF->mImuCalib.Tcb.rowRange(0,3).colRange(0,3));
    Rbc[0] = Rcb[0].transpose();
    tbc[0] = Converter::toVector3d(pF->mImuCalib.Tbc.rowRange(0,3).col(3));
    bf = pF->mbf;

    // if(num_cams>1)
    // {
    //     Eigen::Matrix4d Trl = Converter::toMatrix4d(pF->mTrl);
    //     Rcw[1] = Trl.block<3,3>(0,0)*Rcw[0];
    //     tcw[1] = Trl.block<3,3>(0,0)*tcw[0]+Trl.block<3,1>(0,3);
    //     tcb[1] = Trl.block<3,3>(0,0)*tcb[0]+Trl.block<3,1>(0,3);
    //     Rcb[1] = Trl.block<3,3>(0,0)*Rcb[0];
    //     Rbc[1] = Rcb[1].transpose();
    //     tbc[1] = -Rbc[1]*tcb[1];
    //     pCamera[1] = pF->mpCamera2;
    // }

    // For posegraph 4DoF
    Rwb0 = Rwb;
    DR.setIdentity();
}

void ImuCamPose::SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
              const std::vector<Eigen::Vector3d> &_tbc, const double &_bf)
{
    Rbc = _Rbc;
    tbc = _tbc;
    Rcw = _Rcw;
    tcw = _tcw;
    const int num_cams = Rbc.size();
    Rcb.resize(num_cams);
    tcb.resize(num_cams);

    for(int i=0; i<tcb.size(); i++)
    {
        Rcb[i] = Rbc[i].transpose();
        tcb[i] = -Rcb[i]*tbc[i];
    }
    Rwb = Rcw[0].transpose()*Rcb[0];
    twb = Rcw[0].transpose()*(tcb[0]-tcw[0]);

    bf = _bf;
}



void ImuCamPose::Update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];

    // Update body pose
    twb += Rwb*ut;
    Rwb = Rwb*ExpSO3(ur);

    // Normalize rotation after 5 updates
    its++;
    if(its>=3)
    {
        NormalizeRotation(Rwb);
        its=0;
    }

    // Update camera poses
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;

    Rcw[0] = Rcb[0]*Rbw;
    tcw[0] = Rcb[0]*tbw+tcb[0];

}

void ImuCamPose::UpdateW(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];


    const Eigen::Matrix3d dR = ExpSO3(ur);
    DR = dR*DR;
    Rwb = DR*Rwb0;
    // Update body pose
    twb += ut;

    // Normalize rotation after 5 updates
    its++;
    if(its>=5)
    {
        DR(0,2)=0.0;
        DR(1,2)=0.0;
        DR(2,0)=0.0;
        DR(2,1)=0.0;
        NormalizeRotation(DR);
        its=0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;

    Rcw[0] = Rcb[0]*Rbw;
    tcw[0] = Rcb[0]*tbw+tcb[0];

}


bool VertexPose::read(std::istream& is)
{
    std::vector<Eigen::Matrix<double,3,3> > Rcw;
    std::vector<Eigen::Matrix<double,3,1> > tcw;
    std::vector<Eigen::Matrix<double,3,3> > Rbc;
    std::vector<Eigen::Matrix<double,3,1> > tbc;

    const int num_cams = _estimate.Rbc.size();
    for(int idx = 0; idx<num_cams; idx++)
    {
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                is >> Rcw[idx](i,j);
        }
        for (int i=0; i<3; i++){
            is >> tcw[idx](i);
        }

        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                is >> Rbc[idx](i,j);
        }
        for (int i=0; i<3; i++){
            is >> tbc[idx](i);
        }

        float nextParam;
        is >> nextParam;
    }

    double bf;
    is >> bf;
    _estimate.SetParam(Rcw,tcw,Rbc,tbc,bf);
    updateCache();
    
    return true;
}

bool VertexPose::write(std::ostream& os) const
{
    std::vector<Eigen::Matrix<double,3,3> > Rcw = _estimate.Rcw;
    std::vector<Eigen::Matrix<double,3,1> > tcw = _estimate.tcw;

    std::vector<Eigen::Matrix<double,3,3> > Rbc = _estimate.Rbc;
    std::vector<Eigen::Matrix<double,3,1> > tbc = _estimate.tbc;

    const int num_cams = tcw.size();

    for(int idx = 0; idx<num_cams; idx++)
    {
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                os << Rcw[idx](i,j) << " ";
        }
        for (int i=0; i<3; i++){
            os << tcw[idx](i) << " ";
        }

        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                os << Rbc[idx](i,j) << " ";
        }
        for (int i=0; i<3; i++){
            os << tbc[idx](i) << " ";
        }

    }

    os << _estimate.bf << " ";

    return os.good();
}


VertexVelocity::VertexVelocity(Frame* pF)
{
    setEstimate(Converter::toVector3d(pF->mVw));
}


VertexGyroBias::VertexGyroBias(Frame *pF)
{
    Eigen::Vector3d bg;
    bg << pF->mImuBias.bwx, pF->mImuBias.bwy,pF->mImuBias.bwz;
    setEstimate(bg);
}


VertexAccBias::VertexAccBias(Frame *pF)
{
    Eigen::Vector3d ba;
    ba << pF->mImuBias.bax, pF->mImuBias.bay,pF->mImuBias.baz;
    setEstimate(ba);
}



EdgeInertial::EdgeInertial(IMU::Preintegrated *pInt):JRg(Converter::toMatrix3d(pInt->JRg)),
    JVg(Converter::toMatrix3d(pInt->JVg)), JPg(Converter::toMatrix3d(pInt->JPg)), JVa(Converter::toMatrix3d(pInt->JVa)),
    JPa(Converter::toMatrix3d(pInt->JPa)), mpInt(pInt), dt(pInt->dT)
{
    // This edge links 6 vertices
    resize(6);
    g << 0, 0, -IMU::GRAVITY_VALUE;
    cv::Mat cvInfo = pInt->C.rowRange(0,9).colRange(0,9).inv(cv::DECOMP_SVD);
    Matrix9d Info;
    for(int r=0;r<9;r++)
        for(int c=0;c<9;c++)
            Info(r,c)=cvInfo.at<float>(r,c);
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
     Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}




void EdgeInertial::computeError()
{
    // TODO Maybe Reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const Eigen::Matrix3d dR = Converter::toMatrix3d(mpInt->GetDeltaRotation(b1));
    const Eigen::Vector3d dV = Converter::toVector3d(mpInt->GetDeltaVelocity(b1));
    const Eigen::Vector3d dP = Converter::toVector3d(mpInt->GetDeltaPosition(b1));

    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb
                                                                      - VV1->estimate()*dt - g*dt*dt/2) - dP;
    
    _error << er, ev, ep;
}

void EdgeInertial::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2= static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b1);
    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;

    const Eigen::Matrix3d dR = Converter::toMatrix3d(mpInt->GetDeltaRotation(b1));
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
     // rotation
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; // OK
    _jacobianOplus[0].block<3,3>(3,0) = Skew(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt)); // OK
    _jacobianOplus[0].block<3,3>(6,0) = Skew(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt)); // OK
    // translation
    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity(); // OK

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1; // OK
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt; // OK

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg; // OK
    _jacobianOplus[2].block<3,3>(3,0) = -JVg; // OK
    _jacobianOplus[2].block<3,3>(6,0) = -JPg; // OK

    // Jacobians wrt Accelerometer 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa; // OK
    _jacobianOplus[3].block<3,3>(6,0) = -JPa; // OK

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr; // OK
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2; // OK

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1; // OK
}

EdgeInertialGS::EdgeInertialGS(IMU::Preintegrated *pInt):JRg(Converter::toMatrix3d(pInt->JRg)),
    JVg(Converter::toMatrix3d(pInt->JVg)), JPg(Converter::toMatrix3d(pInt->JPg)), JVa(Converter::toMatrix3d(pInt->JVa)),
    JPa(Converter::toMatrix3d(pInt->JPa)), mpInt(pInt), dt(pInt->dT)
{
    // This edge links 8 vertices
    resize(8);
    gI << 0, 0, -IMU::GRAVITY_VALUE;
    cv::Mat cvInfo = pInt->C.rowRange(0,9).colRange(0,9).inv(cv::DECOMP_SVD);
    Matrix9d Info;
    for(int r=0;r<9;r++)
        for(int c=0;c<9;c++)
            Info(r,c)=cvInfo.at<float>(r,c);
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
     Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}



void EdgeInertialGS::computeError()
{
    // TODO Maybe Reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0],VA->estimate()[1],VA->estimate()[2],VG->estimate()[0],VG->estimate()[1],VG->estimate()[2]);
    g = VGDir->estimate().Rwg*gI;
    const double s = VS->estimate();
   // std::cout<<"s: "<<s<<endl;
    const Eigen::Matrix3d dR = Converter::toMatrix3d(mpInt->GetDeltaRotation(b));
    const Eigen::Vector3d dV = Converter::toVector3d(mpInt->GetDeltaVelocity(b));
    const Eigen::Vector3d dP = Converter::toVector3d(mpInt->GetDeltaPosition(b));
    
    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(s*(VV2->estimate() - VV1->estimate()) - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - g*dt*dt/2) - dP;
    _error << er, ev, ep;
}

void EdgeInertialGS::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0],VA->estimate()[1],VA->estimate()[2],VG->estimate()[0],VG->estimate()[1],VG->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b);

    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d Rwg = VGDir->estimate().Rwg;
    Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3,2);
    Gm(0,1) = -IMU::GRAVITY_VALUE;
    Gm(1,0) = IMU::GRAVITY_VALUE;
    const double s = VS->estimate();
    const Eigen::MatrixXd dGdTheta = Rwg*Gm;
    const Eigen::Matrix3d dR = Converter::toMatrix3d(mpInt->GetDeltaRotation(b));
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
     // rotation
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1;
    _jacobianOplus[0].block<3,3>(3,0) = Skew(Rbw1*(s*(VV2->estimate() - VV1->estimate()) - g*dt));
    _jacobianOplus[0].block<3,3>(6,0) = Skew(Rbw1*(s*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt) - 0.5*g*dt*dt));
    // translation
    _jacobianOplus[0].block<3,3>(6,3) = -s*Eigen::Matrix3d::Identity();

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -s*Rbw1;
    _jacobianOplus[1].block<3,3>(6,0) = -s*Rbw1*dt;

    // Jacobians wrt Gyro bias
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;

    // Jacobians wrt Accelerometer bias
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr;
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = s*Rbw1*Rwb2;

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = s*Rbw1;

    // Jacobians wrt Gravity direction
    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
    _jacobianOplus[6].block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;

    // Jacobians wrt scale factor
    _jacobianOplus[7].setZero();
    _jacobianOplus[7].block<3,1>(3,0) = Rbw1*(VV2->estimate()-VV1->estimate());
    _jacobianOplus[7].block<3,1>(6,0) = Rbw1*(VP2->estimate().twb-VP1->estimate().twb-VV1->estimate()*dt);
}

EdgePriorPoseImu::EdgePriorPoseImu(ConstraintPoseImu *c)
{
    resize(4);
    Rwb = c->Rwb;
    twb = c->twb;
    vwb = c->vwb;
    bg = c->bg;
    ba = c->ba;
    setInformation(c->H);
}

void EdgePriorPoseImu::computeError()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[3]);

    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);
    const Eigen::Vector3d et = Rwb.transpose()*(VP->estimate().twb-twb);
    const Eigen::Vector3d ev = VV->estimate() - vwb;
    const Eigen::Vector3d ebg = VG->estimate() - bg;
    const Eigen::Vector3d eba = VA->estimate() - ba;

    _error << er, et, ev, ebg, eba;
}

void EdgePriorPoseImu::linearizeOplus()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = InverseRightJacobianSO3(er);
    _jacobianOplus[0].block<3,3>(3,3) = Rwb.transpose()*VP->estimate().Rwb;
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(6,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(9,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(12,0) = Eigen::Matrix3d::Identity();
}

void EdgePriorAcc::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

}

void EdgePriorGyro::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

}

// SO3 FUNCTIONS
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    return ExpSO3(w[0],w[1],w[2]);
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return Converter::toMatrix3d(IMU::NormalizeRotation(Converter::toCvMat(res)));
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return Converter::toMatrix3d(IMU::NormalizeRotation(Converter::toCvMat(res)));
    }
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}

Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU()*svd.matrixV();
}
}
