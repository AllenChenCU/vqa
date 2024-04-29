#/bin/bash

for ((i=0; i<15; i+=1)); do
    gcloud compute scp atc2160@instance-cv2-gpu-4-2:/home/atc2160/vqa/logs/mlp_c_only_15epochs_2024_04_29_08_21_29_${i}.pth /Users/allenttchen/columbia/COMS4732/vqa/logs/mlp_c_only_15epochs
done
