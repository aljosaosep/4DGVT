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

#ifndef GOT_INFERENCE_H
#define GOT_INFERENCE_H

#include <boost/program_options.hpp>

// Forward decl.
namespace GOT { namespace tracking { class Hypothesis; }}

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            typedef std::function<double(const GOT::tracking::Hypothesis &, int frame,
                                         const boost::program_options::variables_map &)> f_unary;

            typedef std::function<double(const GOT::tracking::Hypothesis &,
                                         const GOT::tracking::Hypothesis &,
                                         int,
                                         const boost::program_options::variables_map &)> f_pairwise;


            double UnaryFncCAMOT(const GOT::tracking::Hypothesis &hypo, int frame,
                                 const boost::program_options::variables_map &var_map);

            double UnaryFncHypoScore(const GOT::tracking::Hypothesis &hypo, int frame,
                                     const boost::program_options::variables_map &var_map);

            double UnaryFncLogRatios(const GOT::tracking::Hypothesis &hypo, int frame,
                                     const boost::program_options::variables_map &var_map);


            double PairwiseFncIoM(const GOT::tracking::Hypothesis &h1,
                                  const GOT::tracking::Hypothesis &h2,
                                  int frame,
                                  const boost::program_options::variables_map &var_map);

            double PairwiseFncIoU(const GOT::tracking::Hypothesis &h1,
                                  const GOT::tracking::Hypothesis &h2,
                                  int frame,
                                  const boost::program_options::variables_map &var_map);

            f_unary GetUnaryFnc(const std::string &unary_str);

            std::vector<int> InferStateForFrame(int frame,
                                            const std::vector<GOT::tracking::Hypothesis> &hypos,
                                            f_unary unary_fnc,
                                            f_pairwise pairwise_fnc,
                                            const boost::program_options::variables_map &var_map);

        }
    }
}

#endif //GOT_INFERENCE_H
