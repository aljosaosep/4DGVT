/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Francis Engelmann (osep, Engelmann -at- vision.rwth-aachen.de)

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

#ifndef GOT_UTILS_COMMON_H
#define GOT_UTILS_COMMON_H

// std
#include <vector>
#include <map>

// eigen
#include <Eigen/Core>

namespace SUN {
    namespace utils {

        /**
         * @brief Converts a flat index into a tuple of coordinate arrays
        */
        void UnravelIndex(int index, int width, int *x, int *y);

        /**
         * @brief Return 'flattened' index
         */
        void RavelIndex(int x, int y, int width, int *index);

        /**
         * @brief Inverts a pose matrix, does the same as .inv() but this is more efficent
         * @param[in]  pose
         * @param[out] poseInv
         */
        void InvertPose(const Eigen::Matrix4d &pose, Eigen::Matrix4d &pose_inv);

        /**
         * @brief Compute intersection-over-union of tho sets of integers
        */
        double
        InterSectionOverUnionArrays(const std::vector<int> &indices_set_1, const std::vector<int> &indices_set_2);

        /**
         * @brief Parses matrix-string and returns Eigen::MatrixXd.
         * @format Rows are delimited with semi-colons ';', individual numbers with either empty space or colons:
         * @format num00 num01 ... num0k; num10 num11 ... num1k; ...; numl0 numl1 ... numlk;
         * @example "1, 2; 3, 4.2" (2x2 matrix) or "1.0 2.0 3.0; 4.0 5.0 6.0"
        */
        bool ParseMatrixString(const std::string &matrix_str, Eigen::MatrixXd &mat_out);

        /*!
         * @brief Given label_mapper_str, returns one of the pre-defined labels maps.
         */
        std::map<int, std::string> GetCategoryLabelMap(const std::string &label_mapper_str);
    }
}
#endif //GOT_UTILS_COMMON_H
