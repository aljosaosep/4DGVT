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

#ifndef GOT_QPBO_FNC_H
#define GOT_QPBO_FNC_H

#include <vector>
#include <functional>

namespace GOT { namespace tracking { struct Hypothesis; }}
namespace GOT { namespace tracking { struct HypothesisInlier; }}

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            typedef std::function<double(const Hypothesis &, const Hypothesis &, const HypothesisInlier &,
                                         const HypothesisInlier &)> OverlapFncSpecInlier;

            double ExpDecay(int frame, int eval_frame, int tau);

            bool DuplicateTest(const Hypothesis &hypothesis_1, const Hypothesis &hypothesis_2, double thresh_IoU);

            double ComputePhysicalOverlap(int current_frame, int tau, int window_size, const Hypothesis &hypothesis_1,
                                          const Hypothesis &hypothesis_2, OverlapFncSpecInlier overlap_fnc);

            double ComputePhysicalOverlap_IOU_mask(const Hypothesis &hypothesis_1, const Hypothesis &hypothesis_2);


        }
    }
}


#endif //GOT_QPBO_FNC_H
