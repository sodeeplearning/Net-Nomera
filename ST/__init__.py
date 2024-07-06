import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision import transforms
from PIL import Image
from os import listdir
import torchsummary

class Regularization:
    def __init__(self, model):
        self.model = model

    def L1_Regularization(  # Add the result of this function to the loss calculating as follows:
            # loss = <...> + <regularizator name>.L1_Regularization()
            self,
            lamb=1e-4,
    ):
        l1_regularization = 0
        for param in self.model.parameters():
            l1_regularization += torch.norm(param, 1)
        return lamb * l1_regularization

    def L2_Regularization(  # Add the result of this function to the loss calculating as follows:
            # loss = <...> + <regularizator name>.L2_Regularization()
            self,
            lamb=1e-2,
    ):
        l2_regularizaton = 0
        for param in self.model.parameters():
            l2_regularizaton += torch.norm(param, 2)
        return lamb * l2_regularizaton

    def Regularization(
            self,
            level,
            lamb,
    ):
        regularization = 0
        for param in self.model.parameters():
            regularization += torch.norm(param, level)
        return lamb * regularization


class Model_Using(torch.nn.Module):
    def __init__(
            self,
            model,
            device=torch.device('cpu'),
            dtype=torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.Model = model.to(device=self.device)
        self.is_trained = False
        self.regularizator = Regularization(self.Model)

    def Train(
            self,
            train_X,
            train_y,
            val_X,
            val_y,
            num_of_epochs,
            learning_rate,
            batch_size,
            loss_func,
            use_reg=False,
            reg_level=1,
            reg_lamb=1e-4,
            optimizer = 0,
            is_sched_use=True,
            scheduler_freq=500,
            scheduler_gamma=0.5,
            val_every=50,
            show_every=50,
    ):
        self.optimizer = torch.optim.SGD(
            self.Model.parameters(),
            lr=learning_rate
        ) if optimizer == 0 else torch.optim.Adam(self.Model.parameters(), lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=scheduler_gamma)

        self.Model.train()
        losses = {"train": [], "val": []}
        val_batch_size = min(val_X.shape[0], batch_size)

        for epoch in range(1, num_of_epochs + 1):
            print(f"{epoch} epoch now")

            self.optimizer.zero_grad()
            batch = torch.randint(high=train_X.shape[0], size=[batch_size])

            y_pred = self.Model(train_X[batch].to(self.device))
            train_loss = (loss_func(y_pred, train_y[batch]) + self.regularizator.Regularization(
                level=reg_level,
                lamb=reg_lamb,
            ) * use_reg).to(self.device)

            train_loss.backward()
            self.optimizer.step()

            losses["train"].append(train_loss.item())

            if epoch % val_every == 0:
                with torch.no_grad():
                    val_batch = torch.randint(high=val_X.shape[0], size=[val_batch_size])
                    val_pred = self.Model(val_X[val_batch].to(self.device))
                    val_loss = loss_func(val_pred, val_y[val_batch]).to(self.device)

                    losses["val"].append(val_loss.item())

            if epoch % show_every == 0:
                clear_output(True)
                fig, ax = plt.subplots(figsize=(30, 10))
                plt.title("Loss graph")
                plt.plot(losses["train"], ".-", label="Training Loss")
                plt.plot(torch.arange(0, epoch, show_every), losses["val"], ".-", label="Validation Loss")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid()
                plt.show()

            if epoch % scheduler_freq == 0 and is_sched_use:
                self.scheduler.step()

        print("Teaching has been complete successfully")

        self.is_trained = True

    def Use(
            self,
            data
    ):
        self.Model.eval()
        return self.Model(data)

    def Save(
            self,
            file_name,
            cpu_transfer=True
    ):

        transfer_device = torch.device('cpu') if cpu_transfer else self.device
        torch.save(self.Model.to(transfer_device).state_dict(), f"{file_name}.pth ")
        print("The model has been saved successfully")
        return True

    def Load(
            self,
            file_path
    ):
        state_dict = torch.load(file_path)
        self.Model.load_state_dict(state_dict)
        self.Model = self.Model.to(device=self.device, dtype=self.dtype)

        print("The state dict has been loaded successfully")
        return True

    def Info(self, input_shape):
        return torchsummary.summary(self.Model, torch.zeros(input_shape))

    def get_model(self):
        return self.Model


class Residual_Block(torch.nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size,
            activation_func=torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation_func
        padding_size = (kernel_size - 1) // 2

        self.conv_layer = torch.nn.Conv2d(  # Convolution layer
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=padding_size,
        )

        # Residual Connection
        if input_channels != output_channels:
            self.residual_connection = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        else:
            self.residual_connection = torch.Identity()

    def forward(self, x):  # forward propagation functino
        return self.activation(self.conv_layer(x) + self.residual_connection(x))


def getting_files(link):
    return [link + r"/".rstrip() + f for f in listdir(link)]


class Inception_Block(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 # output channels of each conv layer
                 conv_1=3,
                 conv_2=5,
                 conv_3=5,
                 conv_4=3,
                 is_colored=True):
        super().__init__()
        self.input_channels = input_channels

        self.convertation_1 = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=conv_1,
            kernel_size=1,
        )

        self.convertation_2 = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=1,
            kernel_size=1,
        )

        self.convertation_3 = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=1,
            kernel_size=1,
        )

        self.convertation_4 = torch.nn.MaxPool2d(kernel_size=3, stride=1)

        self.convertation_2a = torch.nn.Conv2d(
            in_channels=1,
            out_channels=conv_2,
            kernel_size=3,
            padding=1,
        )

        self.convertation_3a = torch.nn.Conv2d(
            in_channels=1,
            out_channels=conv_3,
            kernel_size=5,
            padding=2,
        )

        self.convertation_4a = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=conv_4,
            kernel_size=1,
            padding=1,
        )

        self.activation = torch.nn.ReLU()

    def forward(self, tensor):
        return self.activation(torch.cat((
            self.convertation_1(tensor),
            self.convertation_2a(self.convertation_2(tensor)),
            self.convertation_3a(self.convertation_3(tensor)),
            self.convertation_4a(self.convertation_4(tensor)),
        ), dim=1))


class GlobalMaxPooling(torch.nn.Module):
    def forward(self, x):
        return x.max(-1).values.max(-1).values


class Conv_Block(torch.nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            activation=torch.nn.ReLU(),
            dropout=0.2,
            padding = 1,
            max_pooling=2,
    ):
        super().__init__()
        self.Block = torch.nn.Sequential(OrderedDict([
            ("Dropout", torch.nn.Dropout(dropout)),
            ("Convolution Layer", torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding = padding,
            )),
            ("Activation Func", activation),
            ("Batch Normalization", torch.nn.BatchNorm2d(output_channels)),
            ("MaxPooling", torch.nn.MaxPool2d(max_pooling))
        ]))

    def forward(self, x):
        return self.Block(x)


class Up_Conv(torch.nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            scale_factor,
            kernel_size=2,
            padding=1,
            stride=1,
            activation_func=torch.nn.ReLU(),
            mode='nearest',
            bias=True,
    ):
        super().__init__()

        self.block = torch.nn.Sequential(OrderedDict([
            ("Upscampling", torch.nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
            )),
            ("Convolution", torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )),
            ("Normalization", torch.nn.BatchNorm2d(output_channels)),
            ("Activation", activation_func),
        ]))

    def forward(
            self,
            tensor
    ):
        return self.block(tensor)


class Conv(torch.nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            activation=torch.nn.ReLU(),
            bias=True,
            dropout = 0.4,
    ):
        super().__init__()

        self.block = torch.nn.Sequential(OrderedDict([
            ("Dropout", torch.nn.Dropout(dropout)),
            ("Convolution", torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )),
            ("Normalization", torch.nn.BatchNorm2d(output_channels)),
            ("Activation", activation),
        ]))

    def forward(
            self,
            tensor
    ):
        return self.block(tensor)


class Down_Conv(torch.nn.Module):
    def __init__(
            self,
            num_of_blocks,
            input_channels,
            output_channels,
            kernel_size=3,
            last_kernel=3,
            pooling_size=2,
            last_padding=1,
            padding=1,
            dropout=0.2,
    ):
        super().__init__()
        conv_blocks = []

        for current_block in range(num_of_blocks):
            conv_blocks.append(Conv(
                input_channels=input_channels if current_block == 0 else output_channels,
                output_channels=output_channels,
                kernel_size=kernel_size if current_block != 0 else last_kernel,
                padding= padding if current_block != 0 else last_padding,
                dropout=dropout,
            ))

        conv_blocks.append(torch.nn.MaxPool2d(pooling_size))

        self.block = torch.nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.block(x)


def jpg_tensor(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(Image.open(image).convert('RGB'))


def imshow(tensor):
    return plt.imshow(tensor.detach().cpu().permute(1, 2, 0))
