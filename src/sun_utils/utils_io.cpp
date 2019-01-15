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

#include "utils_io.h"

// Boost
#include <boost/filesystem.hpp>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

namespace SUN {
    namespace utils {
        namespace IO {
            void WriteEigenMatrixDoubleToTXT(const Eigen::MatrixXd &mat, const char *filename) {
                if (!filename) return;
                std::fstream fileOut(filename, std::fstream::out);
                if (!fileOut.is_open()) return;
                fileOut << mat;
                fileOut.close();
            }

            void WriteEigenMatrixIntToTXT(const Eigen::MatrixXi &mat, const char *filename) {
                if (!filename) return;
                std::fstream fileOut(filename, std::fstream::out);
                if (!fileOut.is_open()) return;
                fileOut << mat;
                fileOut.close();
            }

            void WriteEigenVectorsDoubleToTXT(const std::vector<Eigen::VectorXd> &vectors, const char *filename) {
                if (!filename) return;
                std::ofstream stream_out(filename);
                if (stream_out.is_open()) {
                    for (const auto vec:vectors) {
                        stream_out << vec.transpose() << std::endl;
                    }
                }
                stream_out.close();
            }

            // TODO (Aljosa) add some parse error check!
            bool
            ReadEigenMatrixFromTXT(const char *filename, const int rows, const int cols, Eigen::MatrixXd &mat_out) {
                std::ifstream in_stream(filename);
                if (!in_stream.is_open()) {
                    std::cout << "SUN::utils::IO::Error: Error reading matrix file!" << std::endl;
                    return false;
                }
                mat_out.resize(rows, cols);
                const int mat_size = cols * rows;
                for (int i = 0; i < mat_size; i++) {
                    in_stream >> mat_out(i);
                }
                mat_out.transposeInPlace();
                in_stream.close();
                return true; // Very optimistic!
            }

            bool ReadEigenMatrixFromTXT(const char *filename, Eigen::MatrixXd &mat_out) {

                // General structure
                // 1. Read file contents into vector<double> and count number of lines
                // 2. Initialize matrix
                // 3. Put data in vector<double> into matrix

                std::ifstream input(filename);
                if (input.fail()) {
                    std::cerr << "ReadEigenMatrixFromTXT::Error: Can't Open file:'" << filename << "'." << std::endl;
                    mat_out = Eigen::MatrixXd(0, 0);
                    return false;
                }
                std::string line;
                double d;

                std::vector<double> v;
                int n_rows = 0;
                while (getline(input, line)) {
                    ++n_rows;

                    // Remove delimiters if needed
                    line.erase(std::remove(line.begin(), line.end(), ','), line.end());

                    std::stringstream input_line(line);
                    while (!input_line.eof()) {
                        input_line >> d;
                        v.push_back(d);
                    }
                }
                input.close();

                int n_cols = v.size() / n_rows;
                mat_out = Eigen::MatrixXd(n_rows, n_cols);

                for (int i = 0; i < n_rows; i++)
                    for (int j = 0; j < n_cols; j++)
                        mat_out(i, j) = v[i * n_cols + j];

                return true;
            }

            cv::Mat ReadBinaryDisparityMap(const std::string &filename, unsigned int width, unsigned int height) {
                cv::Mat disparityMap(height, width, CV_32F);
                std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
                if (file.fail()) {
                    std::cout << "Could not read disparity file " << filename << std::endl;
                    return disparityMap;
                }
                float *data = new float[width * height];
                file.read((char *) data, width * height * sizeof(float));
                file.close();

                unsigned int c = 0;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        float d = (float) data[c];
                        disparityMap.at<float>(y, x) = d;
                        c++;
                    }
                }
                return disparityMap;
            }

            void WriteCvMatrixToFile(std::string &path, cv::Mat &mat) {
                std::ofstream myfile;
                myfile.open(path.c_str());
                myfile << mat << std::endl;
                myfile.close();
                return;
            }

            void WriteBinaryDisparityMap(const std::string &filename, cv::Mat &disparityMap) {
                std::ofstream file(filename, std::ios::out | std::ios::binary);
                unsigned int width = disparityMap.cols;
                unsigned int height = disparityMap.rows;
                file.write((char *) disparityMap.data, width * height * sizeof(float));
                file.close();
            }

            bool MakeDir(const char *path) {
                if (!path) {
                    return false;
                }

                boost::filesystem::path fpath(path);
                if (!boost::filesystem::exists(fpath)) {
                    boost::filesystem::path dir(fpath);
                    try {
                        boost::filesystem::create_directories(dir);
                    }
                    catch (boost::filesystem::filesystem_error e) {
                        std::cout << "MakeDir error: " << std::endl << e.what() << std::endl;
                        return false;
                    }
                }
                return true;
            }

            /*
             * Author: Aljosa Osep (osep@vision.rwth-aachen.de)
             */
            bool
            ParseDennisMapFile(char *path, Eigen::Vector4d &ground_plane, Eigen::Matrix3d &K, Eigen::Matrix4d &vo) {

                auto file_handle = std::fopen(path, "r");
                if (!file_handle) {
                    printf("ParseDennisMapFile::Error, could not load data.\r\n");
                    return false;
                }

                char dummy_buff[100];

                // Line 0, 1, 2: Camera int.
                vo.setIdentity();

                float m00, m01, m02, m10, m11, m12, m20, m21, m22;
                std::fscanf(file_handle, "%f %f %f", &m00, &m01, &m02);
                std::fscanf(file_handle, "%f %f %f", &m10, &m11, &m12);
                std::fscanf(file_handle, "%f %f %f", &m20, &m21, &m22);
                K << m00, m01, m02, m10, m11, m12, m20, m21, m22;

                // Skip 3 lines
                std::fgets(dummy_buff, 100, file_handle);
                std::fgets(dummy_buff, 100, file_handle);
                std::fgets(dummy_buff, 100, file_handle);

                // Line 6, 7, 8: R
                std::fscanf(file_handle, "%f %f %f", &m00, &m01, &m02);
                std::fscanf(file_handle, "%f %f %f", &m10, &m11, &m12);
                std::fscanf(file_handle, "%f %f %f", &m20, &m21, &m22);

                // Skip line
                std::fgets(dummy_buff, 100, file_handle);

                // Line 10: T
                float t0, t1, t2;
                std::fscanf(file_handle, "%f %f %f", &t0, &t1, &t2);

                // Skip line
                std::fgets(dummy_buff, 100, file_handle);

                vo << m00, m01, m02, t0,
                        m10, m11, m12, t1,
                        m20, m21, m22, t2,
                        0.0, 0.0, 0.0, 1.0;


                // Line 12: ground-plane
                float a, b, c, d;
                std::fscanf(file_handle, "%f %f %f %f", &a, &b, &c, &d);

                //ground_plane.setZero(4);
                ground_plane << a, b, c, d;

                std::fclose(file_handle);

                return true;
            }

            bool ReadSchipholData(const char *sequence_path, const int frame, cv::Mat &image_rgb, cv::Mat &image_depth,
                                  Eigen::Vector4d &ground_plane, Eigen::Matrix3d &K, Eigen::Matrix4d &vo) {

                /// Make sure path exist
                boost::filesystem::path fpath(sequence_path);
                if (!boost::filesystem::exists(fpath)) {
                    printf("ReadSchipholData::Error, path %s does not exist!\r\n", sequence_path);
                    return false;
                }

                /// Load image and depth map
                char buff[500];
                snprintf(buff, 500, "%s/images/left/rgb_%08d.png", sequence_path, frame);

                if (!boost::filesystem::exists(buff)) {
                    printf("ReadSchipholData error, the file %s does not exist!\r\n", buff);
                    return false;
                }

                image_rgb = cv::imread(buff);

                snprintf(buff, 500, "%s/postproc/dmap/depth_%08d.png", sequence_path, frame);

                if (!boost::filesystem::exists(buff)) {
                    printf("ReadSchipholData error, the file %s does not exist!\r\n", buff);
                    return false;
                }

                image_depth = cv::imread(buff, cv::IMREAD_ANYDEPTH);

                if (!(image_rgb.data && image_depth.data)) {
                    printf("ReadSchipholData::Error, could not load image data!\r\n");
                    return false;
                }

                /// Parse calib. file, extract VO mat. and ground-plane
                snprintf(buff, 500, "%s/postproc/odom/cam_%08d.txt", sequence_path, frame);

                if (!boost::filesystem::exists(buff)) {
                    printf("ReadSchipholData error, the file %s does not exist!\r\n", buff);
                    return false;
                }

                bool parse_dennis_map_success = ParseDennisMapFile(buff, ground_plane, K, vo);
                return parse_dennis_map_success;
            }

            /*
             * Authors: Francis Engelmann (engelmann@vision.rwth-aachen.de), fixed by Aljosa Osep (osep@vision.rwth-aachen.de)
             */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ReadLaserPointCloudKITTI(const std::string file_path) {

                // Allocate 4 MB buffer (only ~130*4*4 KB are needed)
                int32_t num = 1000000;

                FILE *stream;
                stream = fopen(file_path.c_str(), "rb");

                if (!stream) {
                    printf("Could not Open velodyne scan: %s\r\n", file_path.c_str());
                    return nullptr; // Empty
                }

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
                float *data = (float *) malloc(num * sizeof(float));

                // Pointers
                float *px = data + 0;
                float *py = data + 1;
                float *pz = data + 2;
                float *pr = data + 3;

                num = fread(data, sizeof(float), num, stream) / 4;
                for (int32_t i = 0; i < num; i++) {
                    pcl::PointXYZRGBA point;
                    point.x = *px;
                    point.y = *py;
                    point.z = *pz;

                    point.r = 255;
                    point.g = 0;
                    point.b = 0;

                    //double distance_to_velodyn = (point.x*point.x+point.y*point.y+point.z*point.z);
                    //double distance_threshold = 3; // distance to the car/velodyn in m. Within this radius we ignore points as they belong to the recording car
                    // Ignore points closer then threshold, they belong to car.
                    //if (distance_to_velodyn < distance_threshold*distance_threshold) continue;

                    point_cloud->points.push_back(point);
                    px += 4;
                    py += 4;
                    pz += 4;
                    pr += 4;
                }

                free(data);

                fclose(stream);
                return point_cloud;
            }
        }
    }
}



