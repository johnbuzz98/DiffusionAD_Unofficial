dataset_list = 'MVTec VisA'
mvtec_ad_list = 'bottle cable capsule carpet grid hazelnut leather metal nut pill screw tile toothbrush transistor wood zipper'
VisA_list = 'candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum'

for dataset in $dataset_list
do
    if dataset == 'mvtec_ad'
    then
        for class in $mvtec_ad_list
        do
            accelerate launch main.py --yaml_config ./configs/$dataset/$class.yaml
        done
    elif dataset == 'VisA'
    then
        for class in $VisA_list
        do
            accelerate launch main.py --yaml_config ./configs/$dataset/$class.yaml
        done
    fi