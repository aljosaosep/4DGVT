/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dirk Klostermann, Denis Mitzel (osep, mitzel -at- vision.rwth-aachen.de, klostermann -at- rwth-aachen.de)

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


#include <tracking/shape_model.h>

// std
#include <fstream>

// pcl
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

namespace GOT {
    namespace tracking {

        /** ========================================================================
         *      ShapeModel implementation
         * =======================================================================*/

        ShapeModel::PointCloudRGBA::ConstPtr ShapeModel::integrated_points() const {
            return this->integrated_points_;
        }

        double ShapeModel::ProposedOrientation() const {
            return 0;
        }

        std::vector<double> ShapeModel::weights() const {
            return weights_;
        }

        std::vector<double> ShapeModel::variances() const {
            return variances_;
        }

        Eigen::Vector4d ShapeModel::last_centroid_pose() const {
            return this->centroid_poses_.back();
        }

        const std::vector<Eigen::Matrix4d> &ShapeModel::transformations() const {
            return transformations_;
        }
    }
}

