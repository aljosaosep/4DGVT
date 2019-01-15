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

#ifndef GOT_DATASETS_DIRTY_UTILS_H
#define GOT_DATASETS_DIRTY_UTILS_H

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "sun_utils/camera.h"
#include "sun_utils/disparity.h"
#include "utils_kitti.h"

namespace po = boost::program_options;

namespace SUN {
    namespace utils {
        namespace dirty {

            void ComputeDisparityElas(const cv::Mat &image_left, const cv::Mat &image_right,
                                      SUN::DisparityMap &disparity_left, SUN::DisparityMap &disparity_right);

            class DatasetAssitantDirty {

            public:
                DatasetAssitantDirty(const po::variables_map &config_variables_map);

                ~DatasetAssitantDirty();

                bool LoadData__KITTI(int current_frame);

                bool LoadData__OXFORD(int current_frame);

                bool LoadData(int current_frame, const std::string dataset_string);

                bool RequestDisparity(int frame, bool save_if_not_avalible = true);

                void ResetFlags() {
                    // Left / right image
                    got_left_image_ = false;
                    got_right_image_ = false;

                    // Disparity / depth maps
                    got_disparity_map_ = false;

                    // Point clouds
                    got_left_point_cloud_ = false;

                    // Cameras, calib
                    got_left_camera_ = false;
                    got_right_camera_ = false;
                    got_camera_calib_ = false;

                    // Stereo baseline info
                    got_stereo_baseline_ = false;

                    // Ground, odometry info
                    got_ground_plane_ = false;
                }


                po::variables_map variables_map_;

                // Left / right image
                bool got_left_image_ = false;
                bool got_right_image_ = false;

                cv::Mat left_image_;
                cv::Mat right_image_;

                // Disparity / depth maps
                bool got_disparity_map_ = false;
                SUN::DisparityMap disparity_map_;

                // Point clouds
                bool got_left_point_cloud_ = false;
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr left_point_cloud_;


                // Cameras, calib
                bool got_left_camera_ = false;
                bool got_right_camera_ = false;
                bool got_camera_calib_ = false;

                SUN::utils::Camera left_camera_;
                SUN::utils::Camera right_camera_;

                // Stereo baseline info
                bool got_stereo_baseline_ = false;
                double stereo_baseline_;


                // Ground, odometry info
                bool got_ground_plane_ = false;
                Eigen::Vector4d ground_plane_;
            };
        }
    }
}


#endif //GOT_DATASETS_DIRTY_UTILS_H
