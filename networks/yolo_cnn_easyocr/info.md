# Information about this model
This is the first Net Nomer'a model. 
It was made by Vitaliy Petreev [(sodeeplearning)](https://github.com/sodeeplearning) as an experiment for the project
### Model representation
The model represents YOLO, CNN and EasyOCR recognizer (without text detection) applied sequentially.
### Step-by-step explaining:
Initially we have an image. For example: street view.
We pass that image through the YOLO to get bounding boxes of the things that likely to have a number plate.
Then we pass images of these things through a CNN model with 4 outputs to get a bounding box of a number plate.

After detection task we pass images of number plates through the EasyOCR recognizer that represents CNN + RNN to get number's text prediction

#### P.S
So far, this model works not accurately enough, but it will be likely fixed