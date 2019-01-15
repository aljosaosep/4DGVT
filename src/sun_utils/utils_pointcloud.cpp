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

#include "utils_pointcloud.h"

// pcl
#include <pcl/common/transforms.h>

// utils
#include "sun_utils/camera.h"

namespace SUN {
    namespace utils {
        namespace pointcloud {

            void ConvertDisparityMapToPointCloud(
                    const cv::Mat &disparity_map,
                    const cv::Mat &color_image,
                    float c_u,
                    float c_v,
                    float focal_len,
                    float baseline,
                    const Eigen::Matrix4d &pose,
                    const bool withNaN,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud, double far_plane) {

                // Row 0
                const double &p00 = pose(0, 0);
                const double &p01 = pose(0, 1);
                const double &p02 = pose(0, 2);
                const double &p03 = pose(0, 3);

                // Row 1
                const double &p10 = pose(1, 0);
                const double &p11 = pose(1, 1);
                const double &p12 = pose(1, 2);
                const double &p13 = pose(1, 3);

                // Row 2
                const double &p20 = pose(2, 0);
                const double &p21 = pose(2, 1);
                const double &p22 = pose(2, 2);
                const double &p23 = pose(2, 3);

                // Row 3
                const double &p30 = pose(3, 0);
                const double &p31 = pose(3, 1);
                const double &p32 = pose(3, 2);
                const double &p33 = pose(3, 3);

                point_cloud->width = disparity_map.cols;
                point_cloud->height = disparity_map.rows;
                point_cloud->is_dense = false;
                point_cloud->points.resize(disparity_map.cols * disparity_map.rows);

                const double near_plane = 0.1;
                //const double far_plane = 60.0;

                for (int y = 0; y < disparity_map.rows; y++) {
                    for (int x = 0; x < disparity_map.cols; x++) {

                        auto disparity = static_cast<float>(disparity_map.at<float>(y, x));
                        double depth = baseline * focal_len / disparity;
                        if (disparity < 0.001 || disparity >= 99999 || depth >= far_plane || depth <= near_plane) {
                            if (withNaN) {
                                pcl::PointXYZRGBA p;
                                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                                p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                                point_cloud->at(x, y) = p; //->points.push_back(p);
                            }
                            continue;
                        }

                        // Compute 3D point using depth and image-coordinates
                        Eigen::Vector4d point;
                        point[0] = ((double) x - c_u) / focal_len * depth;
                        point[1] = ((double) y - c_v) / focal_len * depth;
                        point[2] = depth;
                        point[3] = 1.0;

                        // Transform point using camera pose
                        point[3] = p30 * point[0] + p31 * point[1] + p32 * point[2] + p33 * point[3];
                        point[0] = (p00 * point[0] + p01 * point[1] + p02 * point[2] + p03 * point[3]) / point[3];
                        point[1] = (p10 * point[0] + p11 * point[1] + p12 * point[2] + p13 * point[3]) / point[3];
                        point[2] = (p20 * point[0] + p21 * point[1] + p22 * point[2] + p23 * point[3]) / point[3];

                        // Pack r/g/b into rgb
                        const auto &intensity = color_image.at<cv::Vec3b>(y, x);

                        // Add point to pointcloud
                        pcl::PointXYZRGBA p;
                        p.x = point(0, 0);
                        p.y = point(1, 0);
                        p.z = point(2, 0);
                        p.r = intensity.val[2];
                        p.g = intensity.val[1];
                        p.b = intensity.val[0];
                        p.a = 255;
                        point_cloud->at(x, y) = p;
                    }
                }
            }

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
            RawLiDARCloudToImageAlignedAndOrganized(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr raw_lidar_cloud,
                                                    const Eigen::Matrix4d &T_lidar_to_cam,
                                                    const cv::Mat &image,
                                                    const SUN::utils::Camera &camera) {

                /// Transform LiDAR cloud to camera space
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr lidar_cloud_cam_space(new pcl::PointCloud<pcl::PointXYZRGBA>);

                pcl::transformPointCloud(*raw_lidar_cloud, *lidar_cloud_cam_space, T_lidar_to_cam);

                /// Make organized image-aligned cloud
                // Take 'visible' portion of the points, append RGB (from image)
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
                tmp_cloud->points.resize(image.rows * image.cols);
                tmp_cloud->height = image.rows;
                tmp_cloud->width = image.cols;
                tmp_cloud->is_dense = false;

                // Init with NaNs
                for (int y = 0; y < image.rows; y++) {
                    for (int x = 0; x < image.cols; x++) {
                        pcl::PointXYZRGBA p;
                        p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                        p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                        tmp_cloud->at(x, y) = p;
                    }
                }

                // Project all LiDAR points to image, append color, add to organized cloud
                for (const auto &pt:lidar_cloud_cam_space->points) {
                    Eigen::Vector4d p_vel(pt.x, pt.y, pt.z, 1.0);
                    if (pt.z < 0) continue;

                    Eigen::Vector3i proj_velodyne = camera.CameraToImage(p_vel);
                    const int u = proj_velodyne[0];
                    const int v = proj_velodyne[1];

                    if (u >= 0 && v >= 0 && u < image.cols && v < image.rows) {
                        auto p_new = pt;
                        p_new.r = static_cast<uint8_t>(image.at<cv::Vec3b>(v, u)[2]);
                        p_new.g = static_cast<uint8_t>(image.at<cv::Vec3b>(v, u)[1]);
                        p_new.b = static_cast<uint8_t>(image.at<cv::Vec3b>(v, u)[0]);
                        p_new.a = 255;
                        tmp_cloud->at(u, v) = p_new;
                    }
                }

                return tmp_cloud;
            }
        }
    }
}
