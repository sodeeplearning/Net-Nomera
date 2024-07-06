import os
import utils

"""
    This is the main script of the project
    
    For contributors:
    If you want to upload your model to the project:
        1) make a directory in networks folder with the name of your model
        2) Past there all your required files
        3) Add a new case with the next number with code like:
            case 2:
                import networks.<your_model_name>'
"""

print("Starting Net Nomer'a...")
print("Choose your neural network model: \n")

available_networks = os.listdir("networks")
for ind, current_network in enumerate(available_networks, start=1):
    print(f"{ind}. {current_network}")

network_index = utils.input_getting(input_type=int,
                                    low_bound=1,
                                    up_bound=len(available_networks))

match network_index:
    case 1:
        import networks.yolo_cnn_easyocr
