import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.optim import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

# Device
USE_CUDA = torch.cuda.is_available()   # check if GPU is available
print("Device : {0}".format("GPU" if USE_CUDA else "CPU"))   # print device type
device = torch.device("cuda" if USE_CUDA else "cpu")   # set device to GPU or CPU based on availability
cpu_device = torch.device("cpu")   # set CPU device

# Train
EPOCHS = 9   # number of training epochs
BATCH_SIZE = 32   # batch size for training
START_EPOCH = 1   # epoch to start training from

lr = 0.0001   # learning rate for optimizer

IMAGE_SIZE = 256   # size of input images
MAX_LEN = 10   # maximum length of captcha text
DATASET_PATH = [   # paths to training dataset
    "/kaggle/input/large-captcha-dataset/Large_Captcha_Dataset",
    "/kaggle/input/captcha-dataset",
    "/kaggle/input/comprasnet-captchas/comprasnet_imagensacerto",
    "/kaggle/input/captcha-images"
]

BAN_DATA = [   # paths to exclude from training dataset
    '/kaggle/input/large-captcha-dataset/Large_Captcha_Dataset/4q2wA.png',
]

RANDOM_SEED = 2004   # random seed for reproducibility
special_char_list = ["<pad>"] # special characters that can appear in captchas (in this case, just padding)
num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']   # numbers that can appear in captchas
upper_alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']   # uppercase letters that can appear in captchas
lower_alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']   # lowercase letters that can appear in captchas

string_list = special_char_list + num_list + upper_alphabet_list + lower_alphabet_list   # list of all characters that can appear in captchas
CHAR_NUM = len(string_list)   # total number of characters that can appear in captchas

token_dictionary = {i : string_list[i] for i in range(len(string_list))}   # dictionary that maps token indices to their corresponding characters
reversed_token_dictionary = {v: k for k, v in token_dictionary.items()}   # dictionary that maps characters to their corresponding token indices

class LACC(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use the EfficientNetV2_S model from torchvision as encoder
        self.encoder = torchvision.models.efficientnet_v2_s().features
        
        # Define the converter parameter which maps encoded features to character embeddings
        self.converter = nn.parameter.Parameter(torch.ones(64, CHAR_NUM))        
        
        # Use SiLU (Sigmoid-Weighted Linear Unit) activation function
        self.silu = nn.SiLU()
        
        # Define fully-connected layers
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 64)
        self.linear3 = nn.Linear(64, MAX_LEN)
        

    def forward(self, x):
        # Pass the input image tensor through the encoder
        feature = self.encoder(x)
        
        # Flatten the feature tensor along dimensions 2 and 3
        feature = torch.flatten(feature, start_dim=2)
        
        # Apply the converter to the flattened feature tensor to obtain character embeddings
        feature = torch.matmul(feature, self.converter)
        
        # Transpose the character embeddings tensor
        y = feature.transpose(-1, -2)
        
        # Pass the transposed tensor through the fully-connected layers, using SiLU activation function in between
        y = self.linear1(y)
        y = self.silu(y)
        y = self.linear2(y)
        y = self.silu(y)
        y = self.linear3(y)
        
        return y

model = LACC().to(device)   # initialize model and move to device
