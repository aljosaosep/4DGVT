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

// boost
#include <boost/tokenizer.hpp>

#include "utils_common.h"
#include "shared_types.h"

namespace SUN {
    namespace utils {

        void UnravelIndex(int index, int width, int *x, int *y) {
            *x = index % width;
            *y = (int) (index / width);
        }


        void RavelIndex(int x, int y, int width, int *index) {
            *index = y * width + x;
        }


        void InvertPose(const Eigen::Matrix4d &pose, Eigen::Matrix4d &pose_inv) {
            Eigen::Vector3d left_position;
            Eigen::Matrix3d R;
            left_position(0) = pose(0, 3);
            left_position(1) = pose(1, 3);
            left_position(2) = pose(2, 3);
            R(0, 0) = pose(0, 0);
            R(0, 1) = pose(0, 1);
            R(0, 2) = pose(0, 2);
            R(1, 0) = pose(1, 0);
            R(1, 1) = pose(1, 1);
            R(1, 2) = pose(1, 2);
            R(2, 0) = pose(2, 0);
            R(2, 1) = pose(2, 1);
            R(2, 2) = pose(2, 2);
            R.transposeInPlace();
            pose_inv.topLeftCorner<3, 3>() = R;
            pose_inv.block<3, 1>(0, 3) = -R * left_position;
            pose_inv(3, 0) = pose_inv(3, 1) = pose_inv(3, 2) = 0.0;
            pose_inv(3, 3) = 1.0;
        }

        double
        InterSectionOverUnionArrays(const std::vector<int> &indices_set_1, const std::vector<int> &indices_set_2) {
            auto inds_p1 = indices_set_1;
            auto inds_p2 = indices_set_2;
            std::sort(inds_p1.begin(), inds_p1.end());
            std::sort(inds_p2.begin(), inds_p2.end());
            std::vector<int> set_intersection(std::max(inds_p1.size(), inds_p2.size()));
            auto it_intersect = std::set_intersection(inds_p1.begin(), inds_p1.end(), inds_p2.begin(), inds_p2.end(),
                                                      set_intersection.begin());
            set_intersection.resize(it_intersect - set_intersection.begin());
            std::vector<int> set_union(inds_p1.size() + inds_p2.size());
            auto it_union = std::set_union(inds_p1.begin(), inds_p1.end(), inds_p2.begin(), inds_p2.end(),
                                           set_union.begin());
            set_union.resize(it_union - set_union.begin());
            const auto IOU = static_cast<double>(set_intersection.size()) / static_cast<double>(set_union.size());
            return IOU;
        }

        bool ParseMatrixString(const std::string &matrix_str, Eigen::MatrixXd &mat_out) {
            boost::char_separator<char> colon_sep(";"); // Row delimiter
            boost::char_separator<char> comma_sep(", "); // Tokens-delimiters (both comma and empty space are ok)
            boost::tokenizer<boost::char_separator<char>> matrix_row_strings(matrix_str, colon_sep);

            // Figure out num cols
            const std::string str = *(matrix_row_strings.begin());
            boost::tokenizer<boost::char_separator<char>> row_tokens(str, comma_sep);
            auto num_rows = std::distance(matrix_row_strings.begin(),
                                          matrix_row_strings.end()); // Num rows ~ num strings
            auto num_cols = std::distance(row_tokens.begin(),
                                          row_tokens.end()); // Check the first row to infer num cols
            mat_out.setZero(num_rows, num_cols);
            int row_num = 0;

            for (auto it = matrix_row_strings.begin(); it != matrix_row_strings.end(); ++it) {
                const std::string matrix_row = *it;
                boost::tokenizer<boost::char_separator<char>> row_tokens(matrix_row, comma_sep);
                auto num_tokens = std::distance(row_tokens.begin(), row_tokens.end());

                // Num tokens in all rows must be equal, otherwise no valid matrix was specified.
                if (num_tokens != num_cols) {
                    printf("Matrix parse error: row_tokens.size()!=num_cols. You sure you specified correct NxM matrix?\r\n");
                    assert(false);
                    return false;
                }

                int col_num = 0;
                for (auto token_it = row_tokens.begin(); token_it != row_tokens.end(); ++token_it) {
                    try {
                        mat_out(row_num, col_num) = std::stof(*token_it);
                    } catch (const std::invalid_argument &e) {
                        printf("Matrix parse error: String to number conversion failed, check that your mat only contains valid floats and delimiters.\r\n");
                        return false;
                    }
                    col_num++;
                }
                row_num++;
            }
            return true;
        }

        std::map<int, std::string> GetCategoryLabelMap(const std::string &label_mapper_str) {
            std::map<int, std::string> ret_map;
            if (label_mapper_str == "coco") {
                ret_map = SUN::shared_types::category_maps::coco_map;
            } else if (label_mapper_str == "kitti") {
                ret_map = SUN::shared_types::category_maps::kitti_map;
            } else {
                printf("Error, invalid label_map str specified!\r\n");
                assert (false);
                return ret_map;
            }

            return ret_map;
        }
    }
}

