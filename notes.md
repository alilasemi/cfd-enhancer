# Creating a Google Cloud Instance
- Go to "Compute Engine" and click "Create Instance".
- Select "Marketplace", search "deep learning" and click on Google's Deep
  Learning VM. Click "Launch".
- Set zone to "northamerica-northeast1-c". Change machine type to GPU. Change
  framework to PyTorch 2.0. Check box for installing NVIDIA drivers. Click
  "Deploy".
- Run this command to set up SSH and set the hostname to "google":
    gcloud compute config-ssh --remove ; gcloud compute config-ssh ; sed 's/Host deeplearning.*/Host google/g' ~/.ssh/config -i
- Run this command to setup SSHfS:
    mkdir -p ~/mnt/google/ ; sshfs -o follow_symlinks,allow_other,reconnect,ServerAliveInterval=15 google:/ ~/mnt/google/

# Run This on the Google Cloud Instance
    pip install transformers accelerate datasets
    git clone https://github.com/huggingface/diffusers ; cd diffusers ; git checkout 1477865e48 ; pip install . ; cd ..
    mkdir output/
    accelerate config

# Transferring Files
    rsync run.sh train_text_to_image_lora.py google:
    rsync main.py coarse.png google:

# Teardown
- Delete Google Cloud instance.
- Run this command to unmount:
    fusermount -u ~/mnt/google/ ; rmdir ~/mnt/google/
