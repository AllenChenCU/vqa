# COMS4732 Computer Vision II Final Project

Project Title: Exploratory models for Visual Question Answering

UNI #: atc2160

Name: Allen Chen

# How to run
```
# Try all zones to get GPUs on GCP
python3 create_gcp_instance.py --project_id="<your project id>" --vm_name="<your vm name>" --gpu_count=4

# pip install
pip install -r requirements.txt

# download abstract scenes and multiple choice data from VQA challenge website
./download.sh

# Preprocess images to get image embeddings
python3 preprocess_images.py

# Train
mkdir logs
python3 train.py --name <run_id>
```

# Results
- see results.ipynb
