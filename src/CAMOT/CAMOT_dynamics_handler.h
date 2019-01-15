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

#ifndef GOT_CAMOT_DYNAMICS_HANDLER_H
#define GOT_CAMOT_DYNAMICS_HANDLER_H

// tracking
#include <tracking/dynamics_handler.h>

// Forward declarations
namespace GOT { namespace tracking { class Hypothesis; }}
namespace SUN { namespace utils { class Camera; }}
namespace GOT { namespace tracking { class Observation; }}

namespace GOT {
    namespace tracking {

        class CAMOTDynamicsHandler : public DynamicsModelHandler {
        public:
            CAMOTDynamicsHandler(const po::variables_map &params);

            void
            InitializeState(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo);

            void ApplyTransition(const SUN::utils::Camera &camera, const Eigen::VectorXd &u, Hypothesis &hypo);

            void
            ApplyCorrection(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo);

            void InitializeMatrices();

        private:
            Eigen::Vector2d G_generic_velocity_variance_;
            Eigen::Vector3d G_generic_pos_variance_;
            Eigen::MatrixXd G_generic_;

            int state_dim_;
        };
    }
}

#endif
