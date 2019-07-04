import torch


model_list = ['I3D', 'S3D_P', 'SS3D_P', 'GSS3D_P']
depth_model_save_dir = './NTU_depth/'
rgb_model_source_dir = './NTU/'
type = 'cv'


for model in model_list:
    source_dir = rgb_model_source_dir + '{}_{}_rgb.pkl'.format(model, type)
    save_dir = depth_model_save_dir + '{}_{}_depth.pkl'.format(model, type)
    rgb_stat_dict = torch.load(source_dir)
    rgb_stat_dict['module.features.0.conv.weight'] = rgb_stat_dict['module.features.0.conv.weight'].sum(1, keepdim=True)
    torch.save(rgb_stat_dict, save_dir)
    print('done')