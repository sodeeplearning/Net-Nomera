import torch
import ST
import pathlib

def Save (object, name):
    dir_path = pathlib.Path().resolve().__str__()
    torch.save(object, dir_path + "/Saved Tensors/" + name + ".pth")

def multy_image(image, multy_factor): # Increasing dataset size in multy_factor times
    augmentation = ST.transforms.Compose([
        ST.transforms.ToPILImage(),
        ST.transforms.ColorJitter(
            brightness=0.4,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        ST.transforms.ToTensor()
    ])

    answer = torch.zeros((multy_factor, 3, 270, 480))

    for current_augmentation in range(multy_factor):
        answer[current_augmentation] = augmentation(image)

    return answer

def augment_dataset(multy_factor, num_of_images = 202):
    images_tensor = torch.load("Saved Tensors/Preaugmented images.pth")

    train_images = torch.zeros((0, 3, 270, 480))
    for ind, current_image in enumerate(images_tensor):
        train_images = torch.cat([train_images, multy_image(
            image = current_image,
            multy_factor = multy_factor
        )], dim = 0)

    Save(train_images, "Augmented images")
    print("Augmenting performed successfully")
