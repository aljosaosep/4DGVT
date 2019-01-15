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

#ifndef GOT_BATCH_PROC_H
#define GOT_BATCH_PROC_H

// Boost
#include <boost/program_options.hpp>

namespace SUN { namespace utils { class Camera; } }
namespace GOT { namespace tracking { class Hypothesis; } }

namespace GOT {
    namespace tracking {

        struct TrackletsResult {
            std::vector<GOT::tracking::Hypothesis> tracklets_;
            std::map<int, SUN::utils::Camera> cameras_;
            int num_frames_processed_;

            TrackletsResult(const std::vector<GOT::tracking::Hypothesis> &tr, const std::map<int, SUN::utils::Camera>& cam_map, int num_proc) {
                tracklets_ = tr;
                cameras_ = cam_map;
                num_frames_processed_ = num_proc;
            }

        };

        namespace CAMOT_tracker {
            TrackletsResult GenerateTracks(int start_frame,
                                           int end_frame,
                                           const boost::program_options::variables_map &variables_map,
                                           const std::string &output_dir,
                                           int debug_level = 0);

            std::map<int, std::vector<int> > RunMAP(int start_frame, int end_frame, const std::vector<GOT::tracking::Hypothesis>& tracklets,
                                                    const po::variables_map &variables_map,
                                                    f_unary unary_fnc,
                                                    f_pairwise pairwise_fnc,
                                                    int debug_level = 0);
        }
    }
}

#endif //GOT_BATCH_PROC_H
