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

#ifndef GOT_KALMAN_FILTER_CONST_VELOCITY_CONST_ELEVATION_H
#define GOT_KALMAN_FILTER_CONST_VELOCITY_CONST_ELEVATION_H

#include <tracking/extended_kalman_filter.h>

namespace GOT {
    namespace tracking {

        class ConstantVelocitySize3DKalmanFilter : public ExtendedKalmanFilter {
        public:
            struct Parameters : public KalmanFilter::Parameters {
                double dt; // Time between two measurements

                Parameters() : KalmanFilter::Parameters(), dt(-100.0) {

                }
            };

            ConstantVelocitySize3DKalmanFilter(const Parameters &params);

            void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0);

            Eigen::VectorXd NonlinearStateProjection() const;

            Eigen::MatrixXd LinearizeTransitionMatrix() const;

            Eigen::VectorXd compute_measurement_residual(const Eigen::VectorXd z_t, const Eigen::MatrixXd &H);

            // Interface
            Eigen::Vector3d GetSize3d() const;

            Eigen::Vector2d GetPoseGroundPlane() const;

            Eigen::Vector2d GetVelocityGroundPlane() const;

            Eigen::Matrix2d GetPoseCovariance() const;

            Eigen::Matrix3d GetSizeCovariance() const;

            Eigen::Matrix2d GetVelocityCovariance() const;

            // Only for this class!
            Eigen::Vector3d GetFullPoseGroundPlane() const;

            Parameters params_;
        };
    }
}

#endif
