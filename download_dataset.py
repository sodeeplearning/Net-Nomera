import io
import zipfile
import requests
import torch

import ST

print("Downloading has been started")

url = "https://storage.yandexcloud.net/net-nomer-dataset/Net-Nomer-a-data_processing.zip"

response = requests.get(url)
zip = zipfile.ZipFile(io.BytesIO(response.content))
zip.extractall()

print("The dataset has been downloaded")
print("Starting detection dataset processing")
def string_perf(string):
    return int(string[string.find('>') + 1: string.rfind('<')])


def xml_perf(path):
    with open(path, 'r') as f:
        file = f.read()
        massive = file.split('\n')
        return (string_perf(massive[-6]) / 1552 * 270,
                string_perf(massive[-7]) / 2592 * 480,
                string_perf(massive[-4]) / 1552 * 270,
                string_perf(massive[-5]) / 2592 * 480)


def get_data(path):
    massive = ST.getting_files(path)
    bbox_tensor = torch.zeros((len(massive) * 5, 100, 4))
    class_tensor = torch.zeros((len(massive) * 5, 100, 2))

    for ind, current_file in enumerate(massive):
        for current_ind in range(5):
            bbox_tensor[ind * 5 + current_ind][0] = torch.tensor(xml_perf(current_file))
            class_tensor[ind * 5 + current_ind][0][1] = 1

    return bbox_tensor, class_tensor


bbox_tensor, class_tensor = get_data('Dataset/boxes')
torch.save(bbox_tensor, 'Saved Tensors/bboxes.pth')
torch.save(class_tensor, 'Saved Tensors/classes.pth')

print("The dataset has been performed")
