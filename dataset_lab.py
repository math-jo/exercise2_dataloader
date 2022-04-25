import os
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode as im
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, path_to_imgs, path_to_sem_masks):
        ''' 
        ***TODO***
        get the file paths to images and masks
        '''
  
        self.path_to_imgs = path_to_imgs
        self.path_to_sem_masks = path_to_sem_masks

        ''' 
        ***TODO***
        list all files you find in the paths above
        '''
        print(1)
        self.images = os.listdir(self.path_to_imgs)
        self.masks = os.listdir(self.path_to_sem_masks)

        # sorts all files
        self.images.sort()
        self.masks.sort()

        # check that we find the same amount of images and masks
        assert len(self.images) == len(self.masks)
       

    def __getitem__(self, index):
        ''' 
        ***TODO***
        extract the element of the list you created above using 'index'
        '''
        
        image = self.images[index]
        sem_mask = self.masks[index]
        ''' 
        ***TODO***
        read image and mask
        WARNING: you might need to resize them to the same dimension, 
                 and you need to convert them to tensors of the right shape 
                 (C x H x W for images, H x W for masks)
        '''
        image = Image.open(self.path_to_imgs + image)
        sem_mask = Image.open(self.path_to_sem_masks + sem_mask)


        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        image = transform(image)
        sem_mask = transform(sem_mask)

        print(image.size())
        print(sem_mask.size())


        return image, sem_mask

    def __len__(self):
        '''
        ***TODO*** 
        return the length of the dataset
        '''
        length = len(self.images)
        print(length)
        return length


def main():
    '''
    ***TODO***
    fill in the paths with the location where you place the provided dataset
    '''
    path_to_train_images = 'images/'
    path_to_train_masks = 'masks/'
    train_data = CustomDataset(path_to_train_images, path_to_train_masks)
    
    '''
    ***TODO***
    the following 6 lines show how to (manually) iterate on the dataloader and 
    show the first image and the first mask
    comment the following lines and write a *for loop* that allows you to 
    visualize all images and masks sequentially
    '''

    print(len(train_data))

    
    iterator = iter(train_data)
    for i in range(6):
        image, mask = next(iterator)
        plt.imshow(image.permute(1, 2, 0)) # the image in tensor form is 3 x H x W, so we need to permute it to H x W x 3 to visualize it
        plt.show()
        plt.imshow(mask.squeeze())
        plt.show()


if __name__ == '__main__':
    main()