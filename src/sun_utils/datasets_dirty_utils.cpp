/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (osep -at- vision.rwth-aachen.de)

rwth_mot framework is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

rwth_mot framework is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
rwth_mot framework; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "datasets_dirty_utils.h"

// boost
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

// utils
#include "sun_utils/utils_io.h"
#include "sun_utils/utils_kitti.h"
#include "sun_utils/utils_pointcloud.h"
#include "sun_utils/ground_model.h"
#include "sun_utils/utils_observations.h"
#include "sun_utils/shared_types.h"

// pcl
#include <pcl/io/pcd_io.h>

// Elas
#include <libelas/elas.h>

#define MAX_PATH_LEN 1000

namespace SUN {
    namespace utils {
        namespace dirty {

            // -------------------------------------------------------------------------------
            // +++ UTILS  +++
            // -------------------------------------------------------------------------------
            void ComputeDisparityElas(const cv::Mat &image_left, const cv::Mat &image_right,
                                      SUN::DisparityMap &disparity_left, SUN::DisparityMap &disparity_right) {

                cv::Mat grayscale_left, grayscale_right;
                cv::cvtColor(image_left, grayscale_left, cv::COLOR_BGR2GRAY);
                cv::cvtColor(image_right, grayscale_right, cv::COLOR_BGR2GRAY);

                // get image width and height
                int32_t width = grayscale_left.cols;
                int32_t height = grayscale_right.rows;

                // allocate memory for disparity images
                const int32_t dims[3] = {width, height, width}; // bytes per line = width
                float *D1_data = (float *) malloc(width * height * sizeof(float));
                float *D2_data = (float *) malloc(width * height * sizeof(float));

                // process
                libelas::Elas::parameters param;

                // Enable these params if you want output disp. maps to be the same as ones used for KITTI eval.
                param.postprocess_only_left = false;
                //param.add_corners = 1;
                //param.match_texture = 0;

                std::vector<libelas::Elas::support_pt> support_points;
                libelas::Elas elas(param);
                elas.process(grayscale_left.data, grayscale_right.data, D1_data, D2_data, dims, support_points);

                disparity_left = SUN::DisparityMap(D1_data, width, height);
                disparity_right = SUN::DisparityMap(D2_data, width, height);

                free(D1_data);
                free(D2_data);
            }

            // -------------------------------------------------------------------------------
            // +++ DATASET ASSISTANT IMPLEMENTATION +++
            // -------------------------------------------------------------------------------

            DatasetAssitantDirty::DatasetAssitantDirty(const po::variables_map &config_variables_map) {
                this->variables_map_ = config_variables_map;
                stereo_baseline_ = -1;
            }

            bool DatasetAssitantDirty::RequestDisparity(int frame, bool save_if_not_avalible) {

                if (!variables_map_.count("left_disparity_path")) {
                    printf("Error, disparity map path option not specified: left_disparity_path.\r\n");
                    return false;
                }

                char left_disparity_map_buff[MAX_PATH_LEN];
                snprintf(left_disparity_map_buff, MAX_PATH_LEN,
                         this->variables_map_["left_disparity_path"].as<std::string>().c_str(), frame);

                // Try to read-in disparity
                disparity_map_.Read(left_disparity_map_buff);

                // No precomputed disparity, see if we have left and right images. If yes, run matching.
                if (disparity_map_.mat().data == nullptr) {
                    printf("Could not load disparity map: %s, running ELAS ...\r\n", left_disparity_map_buff);

                    if (this->left_image_.data == nullptr || this->right_image_.data == nullptr) {
                        printf("Left and right images not available, aborting stereo estimation.\r\n");
                        return false;
                    }

                    SUN::DisparityMap disparity_left, disparity_right;
                    ComputeDisparityElas(this->left_image_, this->right_image_, disparity_left, disparity_right);
                    disparity_map_ = disparity_left;

                    if (save_if_not_avalible) {
                        printf("Saving disparity map to: %s.\r\n", left_disparity_map_buff);
                        boost::filesystem::path prop_path(left_disparity_map_buff);
                        boost::filesystem::path prop_dir = prop_path.parent_path();
                        if (SUN::utils::IO::MakeDir(prop_dir.c_str())) { // Cache disp. map
                            disparity_left.WriteDisparityMap(std::string(left_disparity_map_buff));
                        }
                    }
                }

                return true;
            }

            bool DatasetAssitantDirty::LoadData(int current_frame, const std::string dataset_string) {
                std::string dataset_str_lower = dataset_string;
                std::transform(dataset_str_lower.begin(), dataset_str_lower.end(), dataset_str_lower.begin(),
                               ::tolower);

                this->ResetFlags();

                bool status = false;
                if (dataset_string == "kitti")
                    status = this->LoadData__KITTI(current_frame);
                else if (dataset_string == "oxford")
                    status = this->LoadData__OXFORD(current_frame);
                else
                    std::cout << "DatasetAssitantDirty error: no such dataset: " << dataset_string << std::endl;
                return status;
            }

            bool DatasetAssitantDirty::LoadData__KITTI(int current_frame) {

                // -------------------------------------------------------------------------------
                // +++ ABSOLUTELY REQUIRED DATA +++
                // -------------------------------------------------------------------------------

                /// KITTI camera calibration
                assert(this->variables_map_.count("calib_path"));
                SUN::utils::KITTI::Calibration calibration;
                const std::string calib_path = this->variables_map_["calib_path"].as<std::string>();
                if (!calibration.Open(calib_path)) {
                    printf("DatasetAssitantDirty error: Can't Open calibration file: %s\r\n", calib_path.c_str());
                    return false;
                } else {
                    this->got_camera_calib_ = true;
                }

                /// Left image
                if (this->variables_map_.count("left_image_path")) {
                    char left_image_path_buff[MAX_PATH_LEN];
                    snprintf(left_image_path_buff, MAX_PATH_LEN,
                             this->variables_map_["left_image_path"].as<std::string>().c_str(), current_frame);
                    if (boost::filesystem::exists(left_image_path_buff)) {
                        left_image_ = cv::imread(left_image_path_buff, cv::IMREAD_COLOR);
                        if (left_image_.data == nullptr) {
                            printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                            return false;
                        }

                        this->got_left_image_ = true;
                    } else {
                        return false; // Directly terminate when left_image path is not valid.
                    }
                }

                /// Right image
                if (this->variables_map_.count("right_image_path")) {
                    char right_image_path_buff[MAX_PATH_LEN];
                    snprintf(right_image_path_buff, MAX_PATH_LEN,
                             this->variables_map_["right_image_path"].as<std::string>().c_str(), current_frame);
                    if (boost::filesystem::exists(right_image_path_buff)) {
                        right_image_ = cv::imread(right_image_path_buff, cv::IMREAD_COLOR);
                        if (right_image_.data == nullptr) {
                            printf("DatasetAssitantDirty error: could not load image: %s\r\n", right_image_path_buff);
                            return false;
                        }

                        this->got_right_image_ = true;
                    }
                }

                /// Init camera and ground-model
                left_camera_.init(calibration.GetProjCam2(), Eigen::Matrix4d::Identity(), left_image_.cols,
                                  left_image_.rows);
                right_camera_.init(calibration.GetProjCam3(), Eigen::Matrix4d::Identity(), left_image_.cols,
                                   left_image_.rows);
                stereo_baseline_ = calibration.b();

                this->got_left_camera_ = true;
                this->got_right_camera_ = true;
                this->got_stereo_baseline_ = true;

                // -------------------------------------------------------------------------------
                // +++ OPTIONAL STUFF +++
                // -------------------------------------------------------------------------------
                /// Disparity map
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud_ptr = nullptr;
                if (this->variables_map_.count("left_disparity_path")) {
                    if (!(this->RequestDisparity(current_frame, true))) {
                        printf("DatasetAssitantDirty error: RequestDisparity failed!\r\n");
                        return false;
                    }

                    this->got_disparity_map_ = true;

                    /// Compute point cloud. Note, that this point cloud is in camera space (current frame).
                    left_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
                    Eigen::Matrix<double, 4, 4> identity_matrix_4 = Eigen::MatrixXd::Identity(4, 4);
                    SUN::utils::pointcloud::ConvertDisparityMapToPointCloud(disparity_map_.mat(), left_image_,
                                                                            calibration.c_u(), calibration.c_v(),
                                                                            calibration.f(), calibration.b(),
                                                                            Eigen::Matrix4d::Identity(), true,
                                                                            left_point_cloud_);
                    this->got_left_point_cloud_ = true;
                    point_cloud_ptr = left_point_cloud_;
                }

                return true;
            }

            bool DatasetAssitantDirty::LoadData__OXFORD(int current_frame) {

                // -------------------------------------------------------------------------------
                // +++ ABSOLUTELY REQUIRED DATA +++
                // -------------------------------------------------------------------------------
                /// Left image
                if (this->variables_map_.count("left_image_path")) {
                    char left_image_path_buff[MAX_PATH_LEN];
                    snprintf(left_image_path_buff, MAX_PATH_LEN,
                             this->variables_map_["left_image_path"].as<std::string>().c_str(), current_frame);
                    if (boost::filesystem::exists(left_image_path_buff)) {
                        left_image_ = cv::imread(left_image_path_buff, cv::IMREAD_COLOR);
                        if (left_image_.data == nullptr) {
                            printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                            return false;
                        }

                        this->got_left_image_ = true;
                    } else {
                        return false; // Directly terminate when left_image path is not valid.
                    }
                }

                /// Right image
                if (this->variables_map_.count("right_image_path")) {
                    char right_image_path_buff[MAX_PATH_LEN];
                    snprintf(right_image_path_buff, MAX_PATH_LEN,
                             this->variables_map_["right_image_path"].as<std::string>().c_str(), current_frame);
                    if (boost::filesystem::exists(right_image_path_buff)) {
                        right_image_ = cv::imread(right_image_path_buff, cv::IMREAD_COLOR);
                        if (right_image_.data == nullptr) {
                            printf("DatasetAssitantDirty error: could not load image: %s\r\n", right_image_path_buff);
                            return false;
                        }

                        this->got_right_image_ = true;
                    }
                }

                /// Init camera and ground-model
                assert(this->variables_map_.count("calib_path"));
                const std::string calib_path = this->variables_map_["calib_path"].as<std::string>();
                auto getIntrinsicMatOxford = [calib_path](char *which_cam, bool &ok) -> Eigen::Matrix3d {
                    ok = true;
                    char buff[500];
                    snprintf(buff, 500, "%s/stereo_wide_%s.txt", calib_path.c_str(), which_cam);

                    if (!boost::filesystem::exists(buff)) {
                        ok = false;
                        return Eigen::Matrix3d();
                    }

                    Eigen::MatrixXd stupid_intrinsics;
                    if (!SUN::utils::IO::ReadEigenMatrixFromTXT(buff, stupid_intrinsics)) {
                        ok = false;
                        return Eigen::Matrix3d();
                    }

                    Eigen::Matrix3d K;
                    K.setIdentity();
                    K(0, 0) = stupid_intrinsics(0, 0); // fx
                    K(1, 1) = stupid_intrinsics(0, 1); // fy
                    K(0, 2) = stupid_intrinsics(0, 2); // cx
                    K(1, 2) = stupid_intrinsics(0, 3); // cx

                    return K;
                };

                float oxford_f = 0.0;
                float oxford_c_u = 0.0;
                float oxford_c_v = 0.0;
                float oxford_b = 0.0;
                if (boost::filesystem::is_directory(calib_path)) {
                    bool got_left, got_right;
                    Eigen::Matrix3d K_left = getIntrinsicMatOxford("left", got_left);
                    Eigen::Matrix3d K_right = getIntrinsicMatOxford("right", got_right);

                    if (got_left && got_right) {
                        this->got_camera_calib_ = true;
                    }

                    oxford_f = (float) K_left(0, 0);
                    oxford_c_u = (float) K_left(0, 2);
                    oxford_c_v = (float) K_left(1, 2);
                    oxford_b = 0.24;

                    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
                    Eigen::Matrix<double, 3, 4> T_left, T_right;
                    T_left.setZero();
                    T_right.setZero();
                    T_left.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                    T_right.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                    T_right(0, 3) = -oxford_b;

                    Eigen::Matrix<double, 3, 4> P_left = K_left * T_left * R;
                    Eigen::Matrix<double, 3, 4> P_right = K_right * T_right * R;

                    // TODO
                    left_camera_.init(P_left, Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                    right_camera_.init(P_right, Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                    stereo_baseline_ = oxford_b;

                    this->got_left_camera_ = true;
                    this->got_right_camera_ = true;
                    this->got_stereo_baseline_ = true;
                } else {
                    printf("Error, in case of OXFORD, calib_path should be a dir containing calib. files.\r\n");
                    return false;
                }

                /// Disparity map
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud_ptr = nullptr;
                if (this->variables_map_.count("left_disparity_path")) {

                    if (!(this->RequestDisparity(current_frame, true))) {
                        printf("DatasetAssitantDirty error: RequestDisparity failed!\r\n");
                        return false;
                    }
                    this->got_disparity_map_ = true;

                    /// Compute point cloud. Note, that this point cloud is in camera space (current frame).
                    left_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
                    Eigen::Matrix<double, 4, 4> identity_matrix_4 = Eigen::MatrixXd::Identity(4, 4);

                    SUN::utils::pointcloud::ConvertDisparityMapToPointCloud(disparity_map_.mat(), left_image_,
                                                                            oxford_c_u, oxford_c_v,
                                                                            oxford_f, oxford_b,
                                                                            Eigen::Matrix4d::Identity(), true,
                                                                            left_point_cloud_);

                    this->got_left_point_cloud_ = true;


                    point_cloud_ptr = left_point_cloud_;
                }

                return true;
            }

            DatasetAssitantDirty::~DatasetAssitantDirty() {
                variables_map_.clear();
                left_image_.release();
                right_image_.release();
                disparity_map_;
                left_point_cloud_.reset();
            }
        }
    }
}