#include "OutPut.h"

namespace VIDO_SLAM
{


std::ostream &operator << (std::ostream& output, const SceneObject& object) {
    output << "SceneObject [pose:\nx: " << object.pose.x <<"\ny: " << object.pose.y;
    output << "\n Velocity:\nx: " << object.velocity.x <<"\ny: " << object.velocity.y;
    output << "\nLabel: " << object.label<< " Label index: " << object.label_index << " tracking ID " << object.tracking_id << " ]";

    return output;
}

OutPut::OutPut(int _id, double _timestamp):
    id(_id),
    timestamp(_timestamp) {
        //init camera pose
        camera_pos_translation.x = 0.0;
        camera_pos_translation.y = 0.0;
        camera_pos_translation.z = 0.0;

        //take rotation
        camera_pos_rotation = (cv::Mat_<float>(3,3) << 0, 0, 0,
                                                       0, 0, 0,
                                                       0, 0, 0);
    }

//I think I do want to copy here
void OutPut::add_scene_object(SceneObject _object) {
    scene_objects.push_back(_object);
}
void OutPut::update_camera_pos(cv::Mat& pos_matrix) {
    //take translation part of matrix
    camera_pos_translation.x = pos_matrix.at<float>(0,3);
    camera_pos_translation.y = pos_matrix.at<float>(2,3);
    camera_pos_translation.z = pos_matrix.at<float>(1,3);

    //take rotation
    camera_pos_rotation = (cv::Mat_<float>(3,3) << pos_matrix.at<float>(0,0), pos_matrix.at<float>(0,1), pos_matrix.at<float>(0,2),
                                                   pos_matrix.at<float>(1,0), pos_matrix.at<float>(1,1), pos_matrix.at<float>(1,2),
                                                   pos_matrix.at<float>(2,0), pos_matrix.at<float>(2,1), pos_matrix.at<float>(2,2));
}

void  OutPut::update_camera_vel(cv::Mat& vel_matrix) {
    //take translation part of matrix
    camera_vel_translation.x = vel_matrix.at<float>(0,3);
    camera_vel_translation.y = vel_matrix.at<float>(1,3);
    camera_vel_translation.z = vel_matrix.at<float>(2,3);

    //take rotation
    camera_vel_rotation = (cv::Mat_<float>(3,3) << vel_matrix.at<float>(0,0), vel_matrix.at<float>(0,1), vel_matrix.at<float>(0,2),
                                                   vel_matrix.at<float>(1,0), vel_matrix.at<float>(1,1), vel_matrix.at<float>(1,2),
                                                   vel_matrix.at<float>(2,0), vel_matrix.at<float>(2,1), vel_matrix.at<float>(2,2));
}

std::vector<SceneObject>& OutPut::get_scene_objects() {
    return scene_objects;
}

const int OutPut::scene_objects_size() {
    return scene_objects.size();
}

SceneObject* OutPut::get_scene_objects_ptr() {
    return scene_objects.data();
}
 

const int& OutPut::get_global_fid() const {
    return global_fid;
}

const int& OutPut::get_id() const {
    return id;
}
const double& OutPut::get_timestamp() const {
    return timestamp;
}


}