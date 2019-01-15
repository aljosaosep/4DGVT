#!/bin/bash

# This script executes the video-proposal generator / tracker
# on sequences of KITTI tracking dataset.
# You can specify which KITTI sequences you want to process (0-20)
# and how many do you want to process in parallel.

# Specify which KITTI sequence to execute. By default, all 20.
SEQUENCES=$(seq 0 8)

# Tracker settings
SEGM_INPUTS=mrcnn_tuned # [ mrcnn_tuned | mprcnn_coco | sharpmask_coco ]
INF_MODEL="4DGVT" # [CAMOT|4DGVT]

# Tracker parameters
INPUT_CONF_THRESH=0.8 # Detection/proposal score threshold. In case it is set to 0.8 or more, you will be only getting confident detections.
MAX_NUM_PROPOSALS=300 # Max. num. of proposals passed to the tracker

# Processing settings
MAX_PROC=4 # How many instances do you want to execute in parallel?

if [ "${INF_MODEL}" == "CAMOT" ]; then
        echo "Using CAMOT model ..."
        CFG="../config/CAMOT.cfg"
elif [ "${INF_MODEL}" == "4DGVT" ]; then
        echo "Using 4DGVT model ..."
	CFG="../config/4DGVT.cfg"
else
        echo "Incorrect model specifier (valid: CAMOT | 4DGVT)"
fi

# Data
EXEC=/home/${USER}/projects/tracking-framework/cmake-build-release/apps/CAMOT
KITTI_PATH=/home/${USER}/data/kitti_tracking
DATA_PATH=${KITTI_PATH}/training
PREPROC_DIR=${KITTI_PATH}/preproc
SEQUENCES_DIR=${DATA_PATH}/image_02/*
PROPOSALS_DIR=${PREPROC_DIR}/cached_proposals_${SEGM_INPUTS}

# Output
OUTPUT_DIR=/tmp/tracker_out/
EVAL_DIR=${OUTPUT_DIR}/eval

DATASET_PARAMS="--dataset kitti --dt 0.1"
PROPOSALS_PARAMS="--proposals_confidence_thresh ${INPUT_CONF_THRESH} \
				--proposals_max_number_of_proposals ${MAX_NUM_PROPOSALS} \
				--do_non_maxima_suppression false"

INFERENCE_PARAMS="--run_inference true"

# Process sequentially all KITTI sequences.
for SEQUENCE in ${SEQUENCES[@]}; do
	# Count images for this sequence
	SEQ_NAME=$(printf "%04d" "$SEQUENCE")
	IM_PATH=${DATA_PATH}/image_02/${SEQ_NAME}/*.png	
	IM_ARRAY=($(ls -d ${IM_PATH}))
	IM_ARR_LEN=${#IM_ARRAY[@]}
	((END_FRAME=10#${IM_ARR_LEN}-2))
	START_FRAME=0

	echo " --- Processing: ${SEQ_NAME} ($START_FRAME -> $END_FRAME) --- "

	# Set-up paths
	GENERIC_FILE_NAME="%06d" # Oxford
	LIMG=${DATA_PATH}/image_02/${SEQ_NAME}/${GENERIC_FILE_NAME}.png
	RIMG=${DATA_PATH}/image_03/${SEQ_NAME}/${GENERIC_FILE_NAME}.png
	LDMP=${PREPROC_DIR}/disparity/${SEQ_NAME}/${GENERIC_FILE_NAME}.png
	PROP=${PROPOSALS_DIR}/${SEQ_NAME}/${GENERIC_FILE_NAME}.json
	CALIB=${DATA_PATH}/calib/${SEQ_NAME}.txt
	JSON=${PREPROC_DIR}/kitti_json/${SEGM_INPUTS}/${SEQ_NAME}/${GENERIC_FILE_NAME}.json

	RUN_STR="$EXEC \
	--config_parameters $CFG \
	--left_image_path $LIMG \
	--right_image_path $RIMG \
	--left_disparity_path $LDMP \
	--object_proposals_path $PROP \
	--segmentations_json_file $JSON \
	--calib_path $CALIB \
	--output_dir ${OUTPUT_DIR} \
	--subsequence ${SEQ_NAME} \
	--start_frame ${START_FRAME} --end_frame ${END_FRAME} \
	--eval_output_dir ${EVAL_DIR} \
	--eval_dir_kitti ${EVAL_DIR}/kitti \
	--unary_fnc ${INF_MODEL} \

	${DATASET_PARAMS} \
	${PROPOSALS_PARAMS} \
	${INFERENCE_PARAMS} \

	--debug_level 3"

	#echo ${RUN_STR}
	${RUN_STR} &
	let div="((SEQUENCE+1))%${MAX_PROC}"
 	if [[ $div -eq 0 ]]; then # This ensoures only 4 jobs are being processed in parallel
		echo "Waiting .. ";
		wait;
 	fi
done

echo "All sequences done!"
