import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage import io
import numpy as np
from PIL import Image
class get_dataset(Dataset):
  def __init__(self,dataset,transforms=None):
    self.fg_bgimage,self.bg_image,self.mask_image,self.depth_image=zip(*dataset)
    self.transform=transforms
 
 
  def __len__(self):
    return (len(self.fg_bgimage))
    
  def __getitem__(self,index):
      if(torch.is_tensor(index)):
        index=index.tolist(index)
      input1=Image.open(self.fg_bgimage[index])
      input1.thumbnail((100,100))
      input1=np.asarray(input1)
      input2=(Image.open(self.bg_image[index]))
      input2=np.asarray(input2)
      output1=((Image.open(self.mask_image[index])))
      output1.thumbnail((100,100))
      output1=np.asarray(output1)
      output2=(Image.open(self.depth_image[index]))
      output2.thumbnail((100,100))
      output2=np.asarray(output2)
      output1=output1.transpose(1,0)/255
      output2=output2.transpose(1,0)
      if(self.transform):
        input1=self.transform[0](input1)
        input2=self.transform[1](input2)
        output1=self.transform[2](output1)
        output2=self.transform[3](output2)
      return input1,input2,output1,output2
