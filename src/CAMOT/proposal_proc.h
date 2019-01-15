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

#ifndef GOT_PROPOSAL_PROC_H
#define GOT_PROPOSAL_PROC_H

// opencv
#include <opencv2/core/mat.hpp>

// boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace GOT { namespace segmentation { class ObjectProposal; }}
namespace SUN { namespace utils { class Camera; }}

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace proposal_utils {
                std::vector<GOT::segmentation::ObjectProposal> GeometricFiltering(const SUN::utils::Camera &ref_camera,
                                                                                  const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                                                                                  const po::variables_map &variables_map);

                std::vector<GOT::segmentation::ObjectProposal>
                ProposalsConfidenceFilter(const std::vector<GOT::segmentation::ObjectProposal> &obj_proposals_in,
                                          double thresh = 0.0);
            }
        }
    }
}


#endif //GOT_PROPOSAL_PROC_H
