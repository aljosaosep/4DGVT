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

#include <tracking/extended_kalman_filter.h>

namespace GOT {
    namespace tracking {

        // -------------------------------------------------------------------------------
        // +++ Implementation: EKF base class +++
        // -------------------------------------------------------------------------------

        ExtendedKalmanFilter::ExtendedKalmanFilter(const Parameters &params)
                : KalmanFilter(params) {
        }

        void ExtendedKalmanFilter::ComputePrediction(const Eigen::VectorXd &u, Eigen::VectorXd &x_prio,
                                                     Eigen::MatrixXd &P_prio) const {

            x_prio = NonlinearStateProjection(); // + u;

            if (x_prio.size() != this->params_.state_vector_dim_) {
                x_prio = Eigen::VectorXd();
                P_prio = Eigen::MatrixXd();
                return;
            }

            x_prio += u;

            // Linearize A by computing Jacc. of the (non-linear) transition matrix
            Eigen::MatrixXd A_jacc = LinearizeTransitionMatrix();

            // Project covariance ahead. Note, that A has been linearized.
            P_prio = A_jacc * P_ * A_jacc.transpose() + G_;
        }

        /// H is the observation model matrix (residual = z_t - H*x)
        void ExtendedKalmanFilter::Correction(const Eigen::VectorXd z_t, const Eigen::MatrixXd &observation_cov_t,
                                              const Eigen::MatrixXd &H) {

            Eigen::MatrixXd H_transposed = H.transpose();

            // Compute Kalman gain
            Eigen::MatrixXd HPH_plus_obs_cov = (H * P_ * H_transposed) + observation_cov_t;
            Eigen::MatrixXd K = P_ * H_transposed * HPH_plus_obs_cov.inverse();
            Eigen::VectorXd measurement_residual = compute_measurement_residual(z_t, H);

            x_ = x_ + K * measurement_residual; //(z_t - H_*x_);

            // Update the covariance
            Eigen::MatrixXd KH = K * H;
            P_ = (Eigen::MatrixXd::Identity(KH.rows(), KH.cols()) - KH) * P_;
        }
    }
}



