/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Francis Engelmann (osep, engelmann -at- vision.rwth-aachen.de)

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

#ifndef SUN_UTILS_IO_H
#define SUN_UTILS_IO_H

// C/C++ includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>

// Boost
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

// Forward declarations
namespace SUN { class DisparityMap; }
namespace SUN { namespace utils { class DetectionLayer; }}
namespace GOT { namespace segmentation { class ObjectProposal; }}

namespace SUN {
    namespace utils {
        namespace IO {
            /**
               * @brief Writes content of eigen matrix to file.
               * @param[in] mat
               * @param[in] filename
               */
            void WriteEigenMatrixDoubleToTXT(const Eigen::MatrixXd &mat, const char *filename);

            void WriteEigenMatrixIntToTXT(const Eigen::MatrixXi &mat, const char *filename);

            /**
               * @brief Writes content of eigen vector to file.
               * @param[in] std::vector of Eigen::VectorXd
               * @param[in] filename
               */
            void WriteEigenVectorsDoubleToTXT(const std::vector<Eigen::VectorXd> &vectors, const char *filename);

            /**
               * @brief Read content from file to eigen (double) matrix.
               * @param[in] filename
               * @param[in] rows
               * @param[in] cols
               * @param[in] mat_out
               * @return Sucess flag.
               */
            bool ReadEigenMatrixFromTXT(const char *filename, const int rows, const int cols, Eigen::MatrixXd &mat_out);

            bool ReadEigenMatrixFromTXT(const char *filename, Eigen::MatrixXd &mat_out);

            /**
             * @brief Reads file containing concatenated float values
             * @param filename[in]
             * @param width[in] - of disparity map
             * @param height[in] - of disparity map
             * @return each pixel in Mat contains float values corresponding to disparity
             * @author Francis (engelmann@vison.rwth-aachen.de)
             */
            cv::Mat ReadBinaryDisparityMap(const std::string &filename, unsigned int width, unsigned int height);

            /**
             * @brief writeBinaryDisparityMap
             * @param filename[in] - the name of the file to be written
             * @param disparityMap[in] - the disparity map to we written
             * @author Francis (engelmann@vison.rwth-aachen.de)
             */
            void WriteBinaryDisparityMap(const std::string &filename, cv::Mat &disparityMap);

            /**
             * @brief mtof matrix to file, save cv:mat to text-file
             * @param path[in] - path where to write the mat
             * @param mat[in] - the mat to be saved/written
             * @author Francis (engelmann@vison.rwth-aachen.de)
             */
            void WriteCvMatrixToFile(std::string &path, cv::Mat &mat);

            /**
               * @brief Creates directory.
               * @param[in] mat
               * @param[in] filename
               * @return Returns true upon successful creation of directory.
               */
            bool MakeDir(const char *path);

            /**
               * @brief Read data, stored in 'schipool dataset' format
               * @param[in] subsequence path
               * @param[out] image-rgb
               * @param[out] image-depth
               * @param[out] ground-plane info
               * @param[out] VO matrix
               * @return Returns true upon success.
               */
            bool ReadSchipholData(const char *sequence_path, const int frame, cv::Mat &image_rgb, cv::Mat &image_depth,
                                  Eigen::Vector4d &ground_plane, Eigen::Matrix3d &K, Eigen::Matrix4d &vo);

            /**
           * @brief Parses map format, that Dennis used for stroing camera, ego and ground-plane information.
           */
            bool ParseDennisMapFile(char *path, Eigen::Vector4d &ground_plane, Eigen::Matrix3d &K, Eigen::Matrix4d &vo);

            /**
           * @brief Read KITTI velodyne data.
           * @author Francis (engelmann@vision.rwth-aachen.de)
           */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ReadLaserPointCloudKITTI(const std::string file_path);
        }
    }
}

#endif
