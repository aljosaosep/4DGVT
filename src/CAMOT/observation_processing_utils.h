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

#ifndef GOT_OBSERVATION_PROCESSING_UTILS_H
#define GOT_OBSERVATION_PROCESSING_UTILS_H

#include <opencv2/core/mat.hpp>
#include <pcl/common/common_headers.h>
#include <scene_segmentation/object_proposal.h>
#include <tracking/observation.h>

namespace GOT { namespace tracking { class Hypothesis; }}
namespace GOT { namespace tracking { class Observation; }}
namespace SUN { namespace utils { namespace KITTI { class TrackingLabel; }}}
namespace SUN { namespace utils { class Camera; }};
namespace GOT { namespace segmentation { class ObjectProposal; }}

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace obs_proc {

                /**
                 * @brief This method transforms segmentation::ObjectProposal representation into tracking::Observation (tracker expects 'observation' representation)
                 */
                Observation::Vector
                ProcessObservations(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                    const GOT::segmentation::ObjectProposal::Vector &proposals,
                                    const SUN::utils::Camera &camera);

                /**
                 * @brief Compute obs. velocities based on the input velocity map.
                 */
                Observation::Vector
                ComputeObservationVelocity(const Observation::Vector &observations,
                                           const cv::Mat &velocity_map, double dt);
            }
        }
    }
}

#endif //GOT_OBSERVATION_PROCESSING_UTILS_H
