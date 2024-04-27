#/bin/bash

for ((i=0; i<25; i+=1)); do
    gcloud compute scp atc2160@instance-cv2-gpu-4:/home/atc2160/vqa/logs/attention_data_25epochs_2024_04_27_03_36_04_${i}.pth /Users/allenttchen/columbia/COMS4732/vqa/logs/attention_data_25epochs
done
