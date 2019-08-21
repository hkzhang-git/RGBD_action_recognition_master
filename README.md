# RGBD_action_recognition_master
code for paper "RGB-D Based Action Recognition with Light-weight 3D Convolutional Networks"

Requirement

matplotlib==3.0.3
torch==1.0.0
opencv_python==4.1.0.25
numpy==1.16.2
torchvision==0.2.2
Pillow==6.1.0
tensorboardX==1.8
tensorflow==1.14.0

In this paper, action recognition datasets NTU_RGBD and NUCLA are used.

TO train the model with optical flow, you need to exract the optical flow at first:
cd ./optical_flow/
# extract optical flow for NTU_RGBD dataset
python optical_flow_main.py 
# extract optical flow for NUCLA dataset
python N_UCLA_optical_flow_main.py

# data preprocess
cd preprocess
python NTU_preprocess_main.py
python N_UCLA_preprocess_main.py

# train the model
cd ./training/
python train_main_adam.py --data_root '/home/hkzhang/Documents/sdb_a/Action_recognition_data/ --data_set 'NTU' --data_type
'rgb' --split_type 'cv' --model 'IST'

# use the pretrained model to inference, all pretrained models are stored in ./prediction/trained_models
cd ./prediction/
python prediction_main.py
