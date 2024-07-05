import io
import zipfile
import requests
import torch

import simpletorch as ST

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

print("The dataset has been performed")

