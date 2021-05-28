#ifndef VDO_SLAM_SCENE_H
#define VDO_SLAM_SCENE_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace VIDO_SLAM
{
    struct SceneObject {
        cv::Point3f pose;
        cv::Point2f velocity;
        double yaw=0;
        int label_index; //this is semantic label
        std::string label;
        int tracking_id;


        SceneObject(const SceneObject& scene_object) :
            pose(scene_object.pose),
            velocity(scene_object.velocity),
            label_index(scene_object.label_index),
            label(scene_object.label),
            tracking_id(scene_object.tracking_id) {}

        SceneObject() {}

        friend std::ostream &operator << (std::ostream& output, const SceneObject& object);
        
    };

    class OutPut {
        
        public:

            OutPut(int _id, double _timestamp);
            OutPut(const OutPut& scene) :
                id(scene.get_id()),
                timestamp(scene.get_timestamp()),
                scene_objects(scene.scene_objects),
                camera_pos_translation(scene.camera_pos_translation),
                camera_pos_rotation(scene.camera_pos_rotation),
                camera_vel_translation(scene.camera_vel_translation),
                camera_vel_rotation(scene.camera_vel_rotation) {}


            void add_scene_object(SceneObject _object);
            void update_camera_pos(cv::Mat& pos_matrix); //should be in form [R | t]
            void update_camera_vel(cv::Mat& vel_matrix); //should be in form [R | t]
            std::vector<SceneObject>& get_scene_objects();
            SceneObject* get_scene_objects_ptr();

            const int scene_objects_size();
            const int& get_global_fid() const;
            const int& get_id() const;
            const double& get_timestamp() const;
            std::vector<SceneObject> scene_objects;
            cv::Point3f camera_pos_translation;
            cv::Mat camera_pos_rotation; //should be 3x3 rotation matrix

            cv::Point3f camera_vel_translation;
            cv::Mat camera_vel_rotation; //should be 3x3 rotation matrix

        private:
            int global_fid;
            int id;
            double timestamp;
    };
    
} // namespace VIDO_SLAM

#endif
