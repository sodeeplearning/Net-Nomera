import pandas as pd
import torch
import pathlib

import ST

letter_digit = {
    'A' : 1,
    'B' : 2,
    'C' : 3,
    'E' : 4,
    'H' : 5,
    'K' : 6,
    'M' : 7,
    'O' : 8,
    'P' : 9,
    'T' : 10,
    'X' : 11,
    'Y' : 12
}

dir_path = pathlib.Path().resolve().__str__()
def answer_perform(tensor): # converting image to segmentation answer (1 - where number is, 0 - where no number)
    answer = torch.zeros((270, 480), dtype=torch.long)
    for y in range(270):
        for x in range(480):
            if round(tensor[0][y][x].item(), 5) == 0.99608:
                answer[y][x] = 1
    return answer

def number_perform(string): # converting string with letters to a tensor
    current_num, current_letter = 0, 0
    letters = torch.zeros(3)
    digits = torch.zeros(6)

    for ind, el in enumerate(string):
        if '0' <= el and el <= '9':
            digits[current_num] = int(el)
            current_num += 1
        else:
            letters[current_letter] = letter_digit[el]
            current_letter += 1

    return digits, letters

def Save (object, name):
    torch.save(object, dir_path + "/Saved Tensors/" + name + ".pth")
def image_process(image):
    image_resizing = ST.transforms.Compose([
        ST.transforms.Resize((270, 480))
    ])
    return image_resizing(ST.jpg_tensor(image))

train_images = torch.zeros((202, 3, 270, 480))
train_numbers = torch.zeros((202, 9), dtype=torch.long)
train_seg_answers = torch.zeros((202, 3, 270, 480))

files_in_folder = ST.getting_files(dir_path  + "/Dataset")

paths_to_images = ST.getting_files(files_in_folder[2]) # getting paths to images in dataset
paths_to_seg_answers = ST.getting_files(files_in_folder[1]) # getting paths to answer boxes
numbers = pd.read_json(files_in_folder[0]) # getting text car numbers

print("Dataset performing has been started")

#making tensor with answers for classification
classification_digits = torch.zeros((202, 6), dtype=torch.long)
classification_letters = torch.zeros((202, 3), dtype=torch.long)

for ind, current_string in enumerate(numbers.values):
    digits, letters = number_perform(current_string[0])

    classification_digits[ind] = digits
    classification_letters[ind] = letters

print("Answers for classification were performed successfully")

#making tensor with preaugmented images
for ind, (image_path, answer_path) in enumerate(zip(paths_to_images[::2], paths_to_images[1::2])): #converting jpg into a tensor
    train_images[ind] = image_process(image_path)
    train_seg_answers[ind] = image_process(answer_path)

print("Preaugmented images were performed successfully")

# making tensor with answers for segmentation
train_answers = torch.zeros((202, 270, 480), dtype=torch.long)

for ind, current_tensor in enumerate(train_seg_answers):
    train_answers[ind] = answer_perform(current_tensor)

print("Answer for segmentation performed successfully")

#Saving tensors
Save(classification_letters, "Classification letters")
Save(classification_digits, "Classification digits")
Save(train_images, "Preaugmented images")
Save(train_answers, "Segmentation answers")

#Augmenting dataset
import Augmenting

Augmenting.augment_dataset(
    multy_factor = 5
)
print("Dataset performing has been complete")