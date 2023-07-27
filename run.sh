#!/bin/bash

dataset_list='MVTec VisA'
MVTec_list='bottle cable capsule carpet grid hazelnut leather metal nut pill screw tile toothbrush transistor wood zipper'
VisA_list='candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum'

for dataset in $dataset_list
do
    if [ "$dataset" == 'MVTec' ]
    then
        for class in $MVTec_list
        do
            # Modify the below line based on your requirements and system setup
            echo "Dataset: $dataset, Class: $class"
            accelerate launch main.py --yaml_config "./configs/$dataset/$class.yaml"
        done
    elif [ "$dataset" == 'VisA' ]
    then
        IFS=', ' read -r -a visA_classes <<< "$VisA_list"
        for class in "${visA_classes[@]}"
        do
            # Modify the below line based on your requirements and system setup
            echo "Dataset: $dataset, Class: $class"
            accelerate launch main.py --yaml_config "./configs/$dataset/$class.yaml"
        done
    fi
done
