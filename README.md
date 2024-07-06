# Net-Nomera (Beta)
Neural network for recognizing car numbers

⚠⚠⚠ NOTE that's only Beta version that can have some problems that will be likely fixed in the subsequent versions.
## Authors:

Main AI developer and a Head of the project: [Vitaliy Petreev](https://github.com/sodeeplearning)

Author of the idea and Python developer: [Stepan Andreev](https://github.com/kaferius)

Dataset creator [Stepan Khozhempo](https://github.com/teleportx)
## Installation Guide
#### For Linux:
```bash
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3.12 python3-pip
git clone https://github.com/sodeeplearning/Net-Nomera
cd Net-Nomera
python -m pip install -r requirements.txt
python main.py
```
#### For Windows:
Install the [Python](https://www.python.org/downloads/windows/).
Install the [Git](https://git-scm.com/download/win).

Then pass these commands through a console:
```bash
git clone https://github.com/sodeeplearning/Net-Nomera
cd "Net-Nomera"
pip install -r requirements.txt
python main.py
```

## Using Guide
1) Launch the project with Installation Guide
2) Choose neural network model from the list
3) What for the model installation
4) Put the image file into the project's directory and enter the name of the file (make sure that everything is correct)
5) Get the prediction. Then you will be shown a picture of a car and a text of the number plate 

## Contributors Guide
If you want to make Net Nomer'a better, you can upload your models into this project following these steps:
1) Make a fork of this repository 
2) Upload the 'main' branch
3) Open main.py and read contributors' information or follow subsequent steps:
4) Make a new folder in the 'networks' directory with the name of your model
5) Paste all required files into created folder
6) Add a new case with the next number of 'network_index' with code like:
    ```python
    match network_index:
   
        <...>
   
        case <num_of_models> + 1:
            import networks.<your_model_name> 
    ```
7) Get pull request