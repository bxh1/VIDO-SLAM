#include "viewer/VidoViewer.h"
#include "viewer/pangolin_header/rendertree.h"
#include "viewer/pangolin_header/shader.h"
#include <vector>
namespace VIDO_SLAM
{
MapViewer::MapViewer(const std::string &model_path,const std::string &font_path,const int width,const int height)
    : model_file_(model_path), running_(true),video_img_changed_(false),width_(width),height_(height) {
  Twc_ = Eigen::Matrix4d::Identity();

  glfont_ = new pangolin::GlFont(font_path, 17.0);
  is_pause_.store(true);
  img_ = new unsigned char[3 * width_ * height_];

}
MapViewer::~MapViewer() {}


void LoadgeometryGpu(pangolin::Geometry &geom,
                                pangolin::RenderNode &root) {
  Eigen::AlignedBox3f total_aabb;
  auto aabb = pangolin::GetAxisAlignedBox(geom);
  total_aabb.extend(aabb);

  auto renderable = std::make_shared<pangolin::GlGeomRenderable>(
      pangolin::ToGlGeometry(geom), aabb);

  pangolin::AxisDirection spin_direction = pangolin::AxisNone;
  auto spin_transform =
      std::make_shared<pangolin::SpinTransform>(spin_direction);

  pangolin::RenderNode::Edge edge = {spin_transform, {renderable, {}}};
  root.edges.emplace_back(std::move(edge));
}


void MapViewer::DrawGround() {
  Eigen::Matrix4d twc;
  {
    std::unique_lock<std::mutex> lock(mutex_pose_);
    twc = Twc_;
  }
  glPushMatrix();
  glMultMatrixd(twc.data());
  glLineWidth(6);
  glColor4f(0.1490f, 0.3490f, 0.6490f,0.4);
  glBegin(GL_QUADS);
  glVertex3f(-6.0f,0, -5.0f);
  glVertex3f(6.0f,0,-5.0f);
  glVertex3f(6.0f,0, 7);
  glVertex3f(-6.0f, 0, 7);

  glEnd();
  glPopMatrix();
}


void MapViewer::DrawTrajectory() {
  std::vector<Eigen::Matrix4d> traj;
  {
    std::unique_lock<std::mutex> lock(mutex_pose_);
    traj = trajectorys_;
  }
  if (traj.size() < 1) {
    return;
  }
  glLineWidth(10);
  glColor4f(1.0, 0.0, 0.3,0.6);
  glBegin(GL_LINE_STRIP);
  for (size_t i = 0; i < traj.size() - 1; i++) {
    glVertex3d(traj[i](0, 3), traj[i](1, 3), traj[i](2, 3));
  }
  glEnd();
}

std::vector<float> MapViewer::Generate3DBoxVert(Eigen::Vector3f& center) {

  float vehicle_length = 5.0;
  float vehicle_width = 1.8;
  float vehicle_height = 1.5;

  Eigen::Vector3f corner1(center.x() - vehicle_width / 2,
                          center.y() - vehicle_height / 2,
                          center.z() - vehicle_length / 2);
  Eigen::Vector3f corner2(center.x() + vehicle_width / 2,
                          center.y() - vehicle_height / 2,
                          center.z() - vehicle_length / 2);
  Eigen::Vector3f corner3(center.x() + vehicle_width / 2,
                          center.y() + vehicle_height / 2,
                          center.z() - vehicle_length / 2);
  Eigen::Vector3f corner4(center.x() - vehicle_width / 2,
                          center.y() + vehicle_height / 2,
                          center.z() - vehicle_length / 2);
  Eigen::Vector3f corner5(center.x() - vehicle_width / 2,
                          center.y() - vehicle_height / 2,
                          center.z() + vehicle_length / 2);
  Eigen::Vector3f corner6(center.x() + vehicle_width / 2,
                          center.y() - vehicle_height / 2,
                          center.z() + vehicle_length / 2);
  Eigen::Vector3f corner7(center.x() + vehicle_width / 2,
                          center.y() + vehicle_height / 2,
                          center.z() + vehicle_length / 2);
  Eigen::Vector3f corner8(center.x() - vehicle_width / 2,
                          center.y() + vehicle_height / 2,
                          center.z() + vehicle_length / 2);

  std::vector<float> verts = {
      corner1.x(),corner1.y(),corner1.z(),  corner2.x(),corner2.y(),corner2.z(),
      corner3.x(),corner3.y(),corner3.z(),  corner4.x(),corner4.y(),corner4.z(),  // FRONT

      corner5.x(),corner5.y(),corner5.z(),  corner6.x(),corner6.y(),corner6.z(),
      corner7.x(),corner7.y(),corner7.z(),  corner8.x(),corner8.y(),corner8.z(),  // BACK

      corner5.x(),corner5.y(),corner5.z(),  corner1.x(),corner1.y(),corner1.z(),
      corner4.x(),corner4.y(),corner4.z(),  corner8.x(),corner8.y(),corner8.z(),  // LEFT

      corner2.x(),corner2.y(),corner2.z(),  corner6.x(),corner6.y(),corner6.z(),
      corner7.x(),corner7.y(),corner7.z(),  corner3.x(),corner3.y(),corner3.z(),  // RIGHT

      corner5.x(),corner5.y(),corner5.z(),  corner6.x(),corner6.y(),corner6.z(),
      corner2.x(),corner2.y(),corner2.z(),  corner1.x(),corner1.y(),corner1.z(),  // TOP

      corner7.x(),corner7.y(),corner7.z(),  corner8.x(),corner8.y(),corner8.z(),
      corner4.x(),corner4.y(),corner4.z(),  corner3.x(),corner3.y(),corner3.z()   // BOTTOM
  };

  return verts;
}


void MapViewer::DrawMapPoints()
{
   std::unique_lock<std::mutex> lock(mutex_point_);
   if (map_points_.empty())
    return;

  // draw map points
  
  glPointSize(1.0f);
  glBegin(GL_POINTS);
  glColor3f(0.0f, 0.2f, 0.8f);

  for (auto frame_points : map_points_) 
      for(auto pw : frame_points){
        glVertex3f(pw.at<float>(0), pw.at<float>(1), pw.at<float>(2));
      }
  glEnd();


}
void MapViewer::DrawObjects()
{
  for(size_t i=0;i<objects_.size();i++){
      cv::Point3f obj_pose = objects_[i].pose;
      Eigen::Matrix4f trans_world2object = Eigen::Matrix4f::Identity();
      //float theta = objects_[i].yaw*3.14159/180.0;
      trans_world2object(0, 3) = obj_pose.x;
      trans_world2object(1, 3) = obj_pose.y+1;
      trans_world2object(2, 3) = obj_pose.z;
    //   trans_world2object(0,0)=cosf(theta);
    //   trans_world2object(0,2)=sinf(theta);
    //   trans_world2object(2,0)=-sinf(theta);
    //   trans_world2object(2,2)=cosf(theta);
      trans_world2object.block(0,0,3,3)=Twc_.block(0,0,3,3).cast<float>();
      glPushMatrix();
      glMultMatrixf(trans_world2object.data());
      glLineWidth(1.0);
      glColor4f(0.0, 0.9, 0.7, 0.8);
      Eigen::Vector3f object_center ={0,-0.75,0};
      std::vector<float> verts = Generate3DBoxVert(object_center);
      glVertexPointer(3, GL_FLOAT, 0, verts.data());
      glEnableClientState(GL_VERTEX_ARRAY);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 24);
      glDisableClientState(GL_VERTEX_ARRAY);

      glEnd();
      glPopMatrix();
      
  }
  
}

void InitDefaultPrag(pangolin::GlSlProgram &default_prog) {
  default_prog.ClearShaders();
  std::map<std::string, std::string> prog_defines;

  prog_defines["SHOW_UV"] = "0";
  prog_defines["SHOW_TEXTURE"] = "0";
  prog_defines["SHOW_COLOR"] = "0";
  prog_defines["SHOW_NORMAL"] = "1";
  prog_defines["SHOW_MATCAP"] = "0";

  default_prog.AddShader(pangolin::GlSlAnnotatedShader,
                         pangolin::default_model_shader, prog_defines);
  default_prog.Link();
}

void MapViewer::Run() {
  pangolin::CreateWindowAndBind("apa_localization_viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  double mViewpointF = 2000;
  double mViewpointX = 0;
  double mViewpointY = -70;
  double mViewpointZ = -10;

  {
    std::unique_lock<std::mutex> lock(mutex_pose_);

    if (trajectorys_.size() > 0) {
      mViewpointX = trajectorys_[0](0, 3);
      mViewpointY = trajectorys_[0](1, 3);
    }
  }

  // Load car model geometry

  // Render tree for holding object position
  pangolin::RenderNode root;
  pangolin::GlSlProgram default_prog;
  InitDefaultPrag(default_prog);
  bool load_model_flag = true;
  pangolin::Geometry geom_to_load;
  try {
    geom_to_load = pangolin::LoadGeometry(model_file_);
  } catch (std::exception &e) {
    std::cout << " Error in loading car model " << std::endl;
    std::cout << e.what() << std::endl;
    load_model_flag = false;
  }

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(175));

  pangolin::Var<bool> menuDrawTrajectory("menu.DrawTrajectory", true, true);
  pangolin::Var<bool> menuDrawAxis("menu.DrawAxis", true, true);
  pangolin::Var<bool> menuDrawGroud("menu.DrawGround", false, true);
  pangolin::Var<bool> menuDrawVehicle("menu.DrawVehicle", true, true);
  pangolin::Var<bool> menuRecord("menu.RECORD_VIDEO", false, false);
  pangolin::Var<bool> menuDrawObjects("menu.DrawObjects", true, true);
  pangolin::Var<bool> menuDrawPoints("menu.DrawPoints", true, true);
  pangolin::Var<bool> menuDrawImage("menu.DisplayImage", true, true);
  pangolin::Var<bool> menuPause("menu.Pause/Run", true, false);
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389,
                                 0.1, 1000),
      pangolin::ModelViewLookAt(mViewpointX , mViewpointY, mViewpointZ,
                                0, 0, 0, 0, -1, 0));

  pangolin::View &d_cam =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0 / 768.0)
          .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  // display image
  pangolin::View &raw_image = pangolin::Display("Image")
      .SetBounds(0.8, 1, 0.7, 1, (float)width_ / (float)height_);
  pangolin::GlTexture raw_imageTexture(width_, height_, GL_RGB, false, 0,
                                        GL_BGR, GL_UNSIGNED_BYTE);
 

  while (running_) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(0, 0, 0, 0.0f);
    if (load_model_flag && menuDrawVehicle) {
      s_cam.Follow(GetCarModelMatrix());
      root.edges.clear();
      LoadgeometryGpu(geom_to_load, root);
      default_prog.Bind();
      pangolin::render_tree(default_prog, root, s_cam.GetProjectionMatrix(),
                              s_cam.GetModelViewMatrix(),
                              /*matcaps.size() ? &matcaps[matcap_index] :*/
                              nullptr);

      default_prog.Unbind();
    }

    if (pangolin::Pushed(menuRecord)) {
      pangolin::DisplayBase().RecordOnRender(
          "ffmpeg:[fps=50,bps=8388608,unique_filename,flip=true]//apa-localization.mp4");
    }

    if (menuDrawAxis) {
      DrawAxis();
    }
    
    if (menuDrawTrajectory) {
      DrawTrajectory();
    }

    if(menuDrawGroud)
       DrawGround();
    if(menuDrawObjects){
      DrawObjects();
    }
    if(menuDrawPoints){
      DrawMapPoints();
    }
    if(menuDrawImage) {
       std::lock_guard<std::mutex> lock(mutex_img_);
       if (video_img_changed_)
         raw_imageTexture.Upload(img_,GL_BGR, GL_UNSIGNED_BYTE);
       video_img_changed_ = false;
       raw_image.Activate();
       glColor3f(1.0, 1.0, 1.0);
       raw_imageTexture.RenderToViewportFlipY();
    }
    if(pangolin::Pushed(menuPause)) {
     is_pause_ = !is_pause_;
    }
    
    pangolin::FinishFrame();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void MapViewer::DisplayDynamicImage(cv::Mat img){
     std::lock_guard<std::mutex> lock(mutex_img_);
     img.convertTo(img,CV_8UC3);
     cv::Mat show_img = img.clone();
     memcpy(img_, show_img.data,
         sizeof(unsigned char) * 3 * show_img.rows * show_img.cols);
     video_img_changed_=true;
 }

void MapViewer::SetCurrentPose(const cv::Mat &Twc) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  {
    for (int i = 0; i < 4; i++)
     for (int j = 0; j < 4; j++){
      Twc_(i, j) = Twc.at<float>(i, j);
     }
     Twc_(1,3)=Twc_(1,3)+1.3;
     trajectorys_.push_back(Twc_);
  }
}

void MapViewer::SetMapPoints(const std::vector<std::vector<cv::Mat>>&mps){
  std::unique_lock<std::mutex> lock(mutex_point_);
  map_points_ = mps;
}

void MapViewer::SetObjects(const std::vector<SceneObject> &objects){
  objects_ = objects;
}
void MapViewer::ForceStop() {
  std::unique_lock<std::mutex> lock(mutex_run_);
  running_ = false;
}

bool MapViewer::GetRunstatus() {
  std::unique_lock<std::mutex> lock(mutex_run_);
  bool status = running_;
  return status;
}

void MapViewer::Reset() {
  // clear trajectorys
  std::unique_lock<std::mutex> lock(mutex_pose_);
  trajectorys_.clear();
}


void MapViewer::DrawAxis() {
  glColor3f(1, 0, 0);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(10, 0, 0);
  glColor3f(0, 1, 0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 10, 0);
  glColor3f(0, 0, 1);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, 10);
  glEnd();
}

pangolin::OpenGlMatrix MapViewer::GetCurrentGLMatrix() {
  Eigen::Matrix4d trans_world2cam = Eigen::Matrix4d::Identity();

  {
    std::unique_lock<std::mutex> lock(mutex_pose_);
    trans_world2cam = Twc_;
  }

  pangolin::OpenGlMatrix twc;
  twc.SetIdentity();
  twc.m[12] = trans_world2cam(0, 3);
  twc.m[13] = trans_world2cam(1, 3);
  twc.m[14] = trans_world2cam(2, 3);
  return twc;
}

pangolin::OpenGlMatrix MapViewer::GetCarModelMatrix() {
  Eigen::Matrix4d trans_world2cam = Eigen::Matrix4d::Identity();

  {
    std::unique_lock<std::mutex> lock(mutex_pose_);
    trans_world2cam = Twc_;
  }
  Eigen::Matrix4d tmp_T = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d tmp_R =
       Eigen::AngleAxisd(M_PI/2 , Eigen::Vector3d::UnitX()).matrix()*
       Eigen::AngleAxisd(M_PI/2 , Eigen::Vector3d::UnitZ()).matrix();
  tmp_T.topLeftCorner(3, 3) = tmp_R;
  trans_world2cam=trans_world2cam*tmp_T;
  pangolin::OpenGlMatrix twc;

  twc.SetIdentity();
  twc.m[12] = trans_world2cam(0, 3);
  twc.m[13] = trans_world2cam(1, 3);
  twc.m[14] = trans_world2cam(2, 3);

  twc.m[0] = trans_world2cam(0, 0);
  twc.m[1] = trans_world2cam(1, 0);
  twc.m[2] = trans_world2cam(2, 0);

  twc.m[4] = trans_world2cam(0, 1);
  twc.m[5] = trans_world2cam(1, 1);
  twc.m[6] = trans_world2cam(2, 1);

  twc.m[8] = trans_world2cam(0, 2);
  twc.m[9] = trans_world2cam(1, 2);
  twc.m[10] = trans_world2cam(2, 2);

  return twc;
}

}