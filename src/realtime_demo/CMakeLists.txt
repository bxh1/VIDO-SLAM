cmake_minimum_required(VERSION 3.0.2)
project(demo)


find_package(catkin REQUIRED COMPONENTS
	rosbag
	roscpp
	tf2_ros
	tf2_sensor_msgs
	image_transport
	flow_net
	mask_rcnn
	mono_depth2
	visualization_msgs
	message_generation
	std_msgs
	cv_bridge
	nav_msgs
	sensor_msgs)

find_package(OpenCV 3.4.4 REQUIRED)
find_package(Pangolin REQUIRED)
catkin_package(
  LIBRARIES ${PROJECT_NAME}
	CATKIN_DEPENDS
	roscpp
	rosbag
	rosbag_storage
	image_transport
	std_msgs
	tf2_ros
	tf2_sensor_msgs
	nav_msgs
	sensor_msgs
	cv_bridge
	flow_net
	mask_rcnn
	mono_depth2
	message_runtime
)

include_directories(
    ${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/../../vido_slam/include/
)


add_executable(${PROJECT_NAME}
		  src/run_vido.cc
		  src/utils.cc
           )



target_link_libraries(${PROJECT_NAME}
		${catkin_LIBRARIES}
		${OpenCV_LIBRARIES}
		${Pangolin_LIBRARIES}
		${PROJECT_SOURCE_DIR}/../../vido_slam/lib/libvido_slam.so
		)

