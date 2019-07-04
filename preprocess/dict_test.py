import json


def load_annotation_data(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


dict_1 = load_annotation_data('./data_list/NTU/cs/depth_test.json')
dict_2 = load_annotation_data('./data_list/NTU/cs/rgb_test.json')
print('done')