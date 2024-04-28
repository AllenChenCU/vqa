#/bin/bash

for ((i=0; i<15; i+=1)); do
    gcloud compute scp atc2160@instance-cv2-gpu-4:/home/atc2160/vqa/logs/text_only_data_15epochs_2024_04_28_00_50_09_${i}.pth /Users/allenttchen/columbia/COMS4732/vqa/logs/qc_only_data_15epochs
done
