import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
import easyocr
import utils
from PIL import Image
import ST
from time import sleep

class CNN_Model (torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.convolutions = torch.nn.Sequential(*[
        ST.Conv_Block(
            input_channels = 3,
            output_channels = 128,
        ),
        ST.Conv_Block(
            input_channels = 128,
            output_channels = 128,
        ),
        ST.Conv_Block(
            input_channels = 128,
            output_channels = 256,
        ),
        ST.Conv_Block(
            input_channels = 256,
            output_channels = 512,
        ),
        ST.Conv_Block(
            input_channels = 512,
            output_channels = 512,
        ),
        ST.Conv_Block(
            input_channels = 512,
            output_channels = 512,
        ),
        torch.nn.Flatten()
    ])

    self.fs = torch.nn.Sequential(*[
        torch.nn.Linear(32768, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 4),
    ])

  def forward(self, input_tensor):
    feature_map = self.convolutions(input_tensor)
    fs_bboxes = self.fs(feature_map)
    return fs_bboxes

class Number_detection:
  def __init__(self,
               path_to_model,
               yolo_version = "yolov9c.pt",
               device = 'cpu'):
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.detection_model = YOLO(yolo_version)
    self.cnn_model = CNN_Model()
    self.cnn_model.load_state_dict(torch.load(path_to_model))
    self.cnn_model.train()

  def cropping_image(self, image, x_min, y_min, x_max, y_max, x_shape, y_shape):
    top = int(y_min * y_shape)
    left = int(x_min * x_shape)
    height = int((y_max - y_min) * y_shape)
    width = int((x_max - x_min) * x_shape)
    return transforms.functional.crop(img = image,
                                      top = top,
                                      left = left,
                                      height = height,
                                      width = width)

  def detection_perform(self, image_path, min_size = 0.17):
    detection_output = self.detection_model(image_path)[0]
    classes_with_number = [2, 3, 5, 7]
    transform = transforms.Resize((512, 512))
    answer_images = torch.zeros((0, 3, 512, 512))
    image_tensor = ST.jpg_tensor(image_path)
    y_shape, x_shape = image_tensor.shape[-2:]
    num_of_preds = 0

    for current_class, (x_min, y_min, x_max, y_max) in zip(detection_output.boxes.cls, detection_output.boxes.xyxyn):
      if current_class.item() in classes_with_number and x_max - x_min >= min_size and y_max - y_min >= min_size:
        adding_image = transform(self.cropping_image(image = image_tensor,
                                                     x_min = x_min,
                                                     y_min = y_min,
                                                     x_max = x_max,
                                                     y_max = y_max,
                                                     x_shape = x_shape,
                                                     y_shape = y_shape)).unsqueeze(0)
        answer_images = torch.cat((answer_images,
                                   adding_image), dim=0)
        num_of_preds += 1
    return answer_images, num_of_preds

  def plot_boxes(self, image, coordinates, color='red'):
    x0, y0, x1, y1 = coordinates
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    ST.imshow(image)
    plt.vlines(x_min, y_min, y_max, color=color)
    plt.vlines(x_max, y_max, y_min, color=color)
    plt.hlines(y_min, x_min, x_max, color=color)
    plt.hlines(y_max, x_max, x_min, color=color)

  def load_weights(self, path_to_weights):
    self.cnn_model.load_state_dict(torch.load(path_to_weights))

  def save_cnn_model(self, saving_path):
    torch.save(self.cnn_model, saving_path)

  def pred_perform(self, images_tensor, answer_bboxes):
    answer_images = []
    pil_transform = transforms.ToPILImage()
    for current_image_tensor, (x_min, y_min, x_max, y_max) in zip(images_tensor, answer_bboxes):
      cropped_image = self.cropping_image(image = current_image_tensor,
                                          x_min = x_min,
                                          y_min = y_min,
                                          x_max = x_max,
                                          y_max = y_max,
                                          x_shape = 1,
                                          y_shape = 1)
      answer_images.append(pil_transform(cropped_image))
    return answer_images

  def IMAGE_PRED (self, image_path, show = True, color = 'red', min_size = 0.17):
    images_tensor, num_of_preds = self.detection_perform(image_path, min_size=min_size)
    answer_bboxes = (self.cnn_model((images_tensor.to(self.device)) % 1)).tolist()
    received_images = self.pred_perform(images_tensor, answer_bboxes)
    return images_tensor, answer_bboxes, received_images
class Number_recognizer:
  def __init__(self,
               detect_model_path,
               languages = ['en'],
               image_dir = 'images',
               yolo_version = "yolov9c.pt",
               use_gpu = False,
               use_detector = False):
    self.text_reader = easyocr.Reader(lang_list = languages,
                                      gpu = use_gpu,
                                      detector = use_detector)
    self.detect_model = Number_detection(path_to_model=detect_model_path,
                                         yolo_version=yolo_version)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    self.images_dir = image_dir
    self.characters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y',
              'a', 'b', 'c', 'e', 'h', 'k', 'm', 'o', 'p', 't', 'x', 'y',
              "а", "в", "с", "е", "н", "к", "м", "о", "р", "т", "х", "у",
              "А", "В", "С", "Е", "Н", "К", "М", "О", "Р", "Т", "Х", "У",
              '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

  def PRED(self, image_path):
    images_tensor, bboxes, received_images = self.detect_model.IMAGE_PRED(image_path)
    answer_output = []

    for ind, current_image in enumerate(received_images):
      current_image.save(os.path.join(self.images_dir, f"image{ind}.jpg"))

    for ind, current_path in enumerate(ST.getting_files(self.images_dir)):
      current_output = self.text_reader.recognize(current_path,
                                                  allowlist = self.characters,
                                                  detail = 0)
      answer_output.append(current_output)

    return images_tensor, bboxes, received_images, answer_output

print("Starting the Model...")

image_getting_message = "Put your image into a project directory and enter it's name\n"
pil_transform = transforms.ToPILImage()
number_recognizer = Number_recognizer(detect_model_path = "networks/yolo_cnn_easyocr/cnn_weights/weights_2300.pt")

while True:
    image_path = utils.input_getting(input_message=image_getting_message,
                                     input_type=str)
    while True:
        flag = False
        try:
            images_tensor, bboxes, received_images, number_texts = number_recognizer.PRED(image_path=image_path)
            for current_image, current_number in zip(images_tensor, number_texts):
                Image._show(pil_transform(current_image))
                print(f"Number for the image: {"".join(current_number)}")
                input("Next?")
            flag = True
        except FileNotFoundError:
            print("The path is not valid")
            image_path = utils.input_getting(input_message=image_getting_message,
                                             input_type=str)
        if flag:
            break
    print("Predictions are over")
    continue_answer = utils.input_getting(input_type=str,
                                          input_message="Would you like to continue?\n",
                                          answers=['yes', 'no'])
    if continue_answer == 'no':
        break

print("See you :)")
sleep(3)
