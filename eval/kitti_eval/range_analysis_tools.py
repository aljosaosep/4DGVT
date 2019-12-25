import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ComputeMODARange(range_eval_result):
    num_bins = len(range_eval_result)
    MODA = np.zeros((num_bins,))
    ranges = np.zeros((num_bins,))

    for bin_idx in range(0, num_bins):
        data = range_eval_result[bin_idx]

        fn = float(data['fn'])
        tp = float(data['tp'])
        fp = float(data['fp'])
        n_gt = tp + fn

        MODA_this_bin = 0

        if n_gt > 0:
            MODA_this_bin = 1.0 - (fn + fp) / n_gt

        MODA[bin_idx] = MODA_this_bin
        ranges[bin_idx] = data['Z']

    return ranges, MODA


def ComputeRecallRange(range_eval_result):
    num_bins = len(range_eval_result)
    recall = np.zeros((num_bins,))
    ranges = np.zeros((num_bins,))

    for bin_idx in range(0, num_bins):
        data = range_eval_result[bin_idx]

        fn = float(data['fn'])
        tp = float(data['tp'])
        n_gt = tp + fn
        recall_this_bin = 0

        if n_gt > 0:
            recall_this_bin = tp / n_gt

        recall[bin_idx] = recall_this_bin
        ranges[bin_idx] = data['Z']

    return ranges, recall


def ComputePrecisionRange(range_eval_result):
    num_bins = len(range_eval_result)
    precision = np.zeros((num_bins,))
    ranges = np.zeros((num_bins,))

    for bin_idx in range(0, num_bins):
        data = range_eval_result[bin_idx]

        fn = float(data['fn'])
        tp = float(data['tp'])
        fp = float(data['fp'])
        prec_this_bin = 0
        n_ret = tp + fp

        if n_ret > 0:
            prec_this_bin = tp / n_ret

        precision[bin_idx] = prec_this_bin
        ranges[bin_idx] = data['Z']

    return ranges, precision

def Compute3Dloc(range_eval_result):
    num_bins = len(range_eval_result)
    loc = np.zeros((num_bins,))
    ranges = np.zeros((num_bins,))

    for bin_idx in range(0, num_bins):
        data = range_eval_result[bin_idx]

        fn = float(data['fn'])
        tp = float(data['tp'])
        #n_gt = tp + fn
        # recall_this_bin = 0
        d3d = 0.0

        if tp > 0:
            #recall_this_bin = tp / n_gt
            d3d = data['dist3d'] / tp
        loc[bin_idx] = d3d
        ranges[bin_idx] = data['Z']

    return ranges, loc