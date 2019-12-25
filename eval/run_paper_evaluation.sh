#!/bin/bash

EVAL_PROPOSALS_OXFORD=false
EVAL_TRACKERS=true

PROPOSALS_EVAL_SCRIPT=eval_single_image_proposals.py
CLEARMOT_EVAL_SCRIPT=evaluate_CLEARMOT.py
RESULTS_DIR=/tmp/prop4d
DATA_DIR=data
#PROPOSAL_DIRS_TO_EVAL_OXFORD=(proposals_and_tracklets oxford_proposal_baselines tracklets)

PROPOSAL_DIRS_TO_EVAL_OXFORD=(proposals_and_tracklets)

#PROPOSAL_DIRS_TO_EVAL_OXFORD=(tracklets)

TRACKING_DIRS_TO_EVAL=(kitti_tracking_baselines)
TUBE_DIRS_TO_EVAL=(proposals_and_tracklets)

# Run proposal evaluation
# Note: script below assumes python 3.x
TSTAMP=`date "+%Y-%m-%d-%H-%M-%S"`

# Proposals - Oxford
if [ "$EVAL_PROPOSALS_OXFORD" = true ]; then
	OUT_DIR=${RESULTS_DIR}/$TSTAMP/proposals/
	mkdir -p ${OUT_DIR}
	python ${PROPOSALS_EVAL_SCRIPT} --plot_output_dir ${OUT_DIR} --evaluate_dir ${DATA_DIR}/proposals --do_not_timestamp
fi


# Run KITTI tracking evaluation
# Note: CLEAR MOT eval script assumes python2.7; trackingannotationtool
cd tracking_eval_kitti
if [ "$EVAL_TRACKERS" = true ]; then
	for TR_DIR in ../data/kitti_tracking_baselines/*; do
		echo "Evaluating trackers in: ${TR_DIR}"
		OUT_DIR=${RESULTS_DIR}/$TSTAMP/trackers/${TR_DIR}
		source activate py27 # Activate my py27 env
		mkdir -p ${OUT_DIR}
		python ${CLEARMOT_EVAL_SCRIPT} --evaluate_dir ${TR_DIR} --plot_output_dir ${OUT_DIR} --seqmap_path evaluate_tracking_paul_split.seqmap --do_not_timestamp
	done
fi
