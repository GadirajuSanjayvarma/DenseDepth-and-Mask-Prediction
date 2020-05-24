from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4modeltrainer import ModelTrainer

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros",stride=1):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode,stride=stride)]
 
    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias,padding=0)]
 
    def activate(self, l, out_channels, bn=True, dropout=0, relu=True,max_pooling=0):
      if(max_pooling>0):
        l.append(nn.MaxPool2d(2,2))
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())
 
      return nn.Sequential(*l)
 
    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="circular",max_pooling=0,stride=1):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode,stride=stride), out_channels, bn, dropout, relu,max_pooling)
 
    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="circular"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)
 
    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name
 
    def summary(self, input_size,input_size1): #input_size=(1, 28, 28)
      summary(self, input_size=[input_size,input_size1])
 
    def gotrain(self,model, optimizer, train_loader, test_loader, epochs, statspath,criterion,writer,scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(model,optimizer, train_loader, test_loader, statspath,criterion,writer,scheduler,batch_scheduler, L1lambda)
      #print("hello")
      self.trainer.run(epochs)
 
    def stats(self):
      return self.trainer.stats if self.trainer else None
 

class depth_mask_model(Net):
  def __init__(self,name="Model",dropout_value=0.0):
    super(depth_mask_model,self).__init__(name)
    #first part of architecture of solving the mask image
    self.conv1=self.create_conv2d(3,16) #receptive field 3
    self.conv2=self.create_conv2d(16,16) #receptive field 5
    self.conv3=self.create_conv2d(16,32) #receptive field 7
    self.conv4=self.create_conv2d(32,32) #receptive field 9
    self.conv5=self.create_conv2d(64,64,dilation=2,padding=2) #receptive field 13
    self.conv6=self.create_conv2d(64,64,dilation=4,padding=4) #receptive field 19
    self.conv7=self.create_conv2d(64,128,dilation=8,padding=8) #receptive field 29
    self.conv8=self.create_conv2d(128,64) #receptive field 31
    self.conv9=self.create_conv2d(64,16,stride=2,padding=1) #receptive field 33
    self.conv10=self.create_conv2d(16,16) #receptive field 37
    self.conv11=self.create_conv2d(16,32,dilation=2,padding=2) #receptive field 45
    self.conv12=self.create_conv2d(32,32,dilation=4,padding=4) #receptive field 57
    self.conv13=self.create_conv2d(32,64,dilation=8,padding=8) #receptive field 77
    self.conv14=self.create_conv2d(64,64,dilation=16,padding=16) #receptive field 113
    self.conv15=self.create_conv2d(64,128,dilation=16,padding=16) #receptive field 149
    self.conv16=self.create_conv2d(128,64,dilation=16,padding=16) #receptive field 185
    self.conv17=self.create_conv2d(64,64) #receptive field 189
    self.upsampling1=nn.Upsample((100,100), mode='nearest') #receptive field 193 formula is 2^(no of sampling layers)*(kernalsize-1)
    self.conv18=self.create_conv2d(128,128) #receptive field 197
    self.conv19=self.create_conv2d(128,1,bn=False, dropout=0, relu=False) #receptive field 201
    
    #operations used by our background image
    self.conv20=self.create_conv2d(3,16) #receptive field 3
    self.conv21=self.create_conv2d(16,16) #receptive field 5
    self.conv22=self.create_conv2d(16,32) #receptive field 7
    self.conv23=self.create_conv2d(32,32) #receptive field 9

    #first part of architecture of solving the depth image
    self.conv_depth1=self.create_conv2d(3,16) #receptive field 3
    self.conv_depth2=self.create_conv2d(16,16) #receptive field 5
    self.conv_depth3=self.create_conv2d(16,32) #receptive field 7
    self.conv_depth4=self.create_conv2d(32,32) #receptive field 9
    self.conv_depth5=self.create_conv2d(64,64,dilation=2,padding=2) #receptive field 13
    self.conv_depth6=self.create_conv2d(64,64,dilation=4,padding=4) #receptive field 19
    self.conv_depth7=self.create_conv2d(64,128,dilation=8,padding=8) #receptive field 29
    self.conv_depth8=self.create_conv2d(128,64) #receptive field 31
    self.conv_depth9=self.create_conv2d(64,16,stride=2,padding=1) #receptive field 33
    self.conv_depth10=self.create_conv2d(16,16) #receptive field 37
    self.conv_depth11=self.create_conv2d(16,32,dilation=2,padding=2) #receptive field 45
    self.conv_depth12=self.create_conv2d(32,32,dilation=4,padding=4) #receptive field 57
    self.conv_depth13=self.create_conv2d(32,64,dilation=8,padding=8) #receptive field 77
    self.conv_depth14=self.create_conv2d(64,64,dilation=16,padding=16) #receptive field 113
    self.conv_depth15=self.create_conv2d(64,128,dilation=16,padding=16) #receptive field 149
    self.conv_depth16=self.create_conv2d(128,64,dilation=16,padding=16) #receptive field 185
    self.conv_depth17=self.create_conv2d(64,64) #receptive field 189
    self.upsampling2=nn.Upsample((100,100), mode='nearest') #receptive field 193 formula is 2^(no of sampling layers)*(kernalsize-1)
    self.conv_depth18=self.create_conv2d(128,128) #receptive field 197
    self.conv_depth19=self.create_conv2d(128,1,bn=False, dropout=0, relu=False) #receptive field 201
    
    #operations used by our background image
    self.conv_depth20=self.create_conv2d(3,16) #receptive field 3
    self.conv_depth21=self.create_conv2d(16,16) #receptive field 5
    self.conv_depth22=self.create_conv2d(16,32) #receptive field 7
    self.conv_depth23=self.create_conv2d(32,32) #receptive field 9

  def forward(self,x1,x2):
    x1.requires_grad=False
    x2.requires_grad=False
    #first part for finding mask
    
    # operations of background image
    bg_image_mask=self.conv20(x2)
    bg_image_mask=self.conv21(bg_image_mask)
    bg_image_mask=self.conv22(bg_image_mask)
    bg_image_mask=self.conv23(bg_image_mask)

    output1=self.conv1(x1)
    output1=self.conv2(output1)
    output1=self.conv3(output1)
    output1=self.conv4(output1)
    output1=torch.cat((output1,bg_image_mask),1)
    output1=self.conv5(output1)
    output1=self.conv6(output1)
    output1=self.conv7(output1)
    output1=self.conv8(output1)
    self.concat1=output1
    output1=self.conv9(output1)
    output1=self.conv10(output1)
    output1=self.conv11(output1)
    output1=self.conv12(output1)
    output1=self.conv13(output1)
    output1=self.conv14(output1)
    output1=self.conv15(output1)
    output1=self.conv16(output1)
    output1=self.conv17(output1)
    output1=self.upsampling1(output1)
    output1=torch.cat((output1,self.concat1),1)
    output1=self.conv18(output1)
    output1=self.conv19(output1)


   # operations on depth image
    bg_image_depth=self.conv_depth20(x2)
    bg_image_depth=self.conv_depth21(bg_image_depth)
    bg_image_depth=self.conv_depth22(bg_image_depth)
    bg_image_depth=self.conv_depth23(bg_image_depth)

    output2=self.conv_depth1(x1)
    output2=self.conv_depth2(output2)
    output2=self.conv_depth3(output2)
    output2=self.conv_depth4(output2)
    output2=torch.cat((output2,bg_image_depth),1)
    output2=self.conv_depth5(output2)
    output2=self.conv_depth6(output2)
    output2=self.conv_depth7(output2)
    output2=self.conv_depth8(output2)
    self.concat2=output2
    output2=self.conv_depth9(output2)
    output2=self.conv_depth10(output2)
    output2=self.conv_depth11(output2)
    output2=self.conv_depth12(output2)
    output2=self.conv_depth13(output2)
    output2=self.conv_depth14(output2)
    output2=self.conv_depth15(output2)
    output2=self.conv_depth16(output2)
    output2=self.conv_depth17(output2)
    output2=self.upsampling2(output2)
    output2=torch.cat((output2,self.concat2),1)
    output2=self.conv_depth18(output2)
    output2=self.conv_depth19(output2)
 
    return output1,output2

class Cfar10Net(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 32, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x32, RF = 3
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 5
        self.conv3 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 16x16x32, OUT 16x16x64, RF = 12
        self.conv5 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 16x16x64, OUT 16x16x64, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv6 = self.create_conv2d(64, 128, dropout=dropout_value) # IN 8x8x64, OUT 8x8x128, RF = 26
        self.conv7 = self.create_conv2d(128, 128, dropout=dropout_value) # IN 8x8x128, OUT 8x8x128, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        
        self.conv8 = self.create_depthwise_conv2d(128, 256, dropout=dropout_value) # IN 4x4x128, OUT 4x4x256, RF = 54
        self.conv9 = self.create_depthwise_conv2d(256, 256, dropout=dropout_value) # IN 4x4x256, OUT 4x4x256, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(256, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net2(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net2, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net3(Net):
    def __init__(self, name="Cfar10Net3", dropout_value=0):
        super(Cfar10Net3, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_depthwise_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_depthwise_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_depthwise_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_depthwise_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_depthwise_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_depthwise_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



class Cfar10Net4(Net):
    def __init__(self, name="Cfar10Net4", dropout_value=0):
        super(Cfar10Net4, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net5(Net):
    def __init__(self, name="Cfar10Net5", dropout_value=0):
        super(Cfar10Net5, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value, dilation=2, padding=2) # IN 16x16x16, OUT 16x16x32, RF = 16
        #self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64, RF = 34
        #self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 70
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 86

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        #x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        #x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class MnistNet(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(MnistNet, self).__init__(name)

        # Input Convolution Block
        self.convblock1 = self.create_conv2d(3, 32, dropout=dropout_value, groups=3) # input_size = 28, output_size = 28, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = self.create_conv2d(10, 10, dropout=dropout_value) # output_size = 28, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6

        self.convblock3 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 12, RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock4 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 10, RF = 14

        self.convblock5 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 8, RF = 18

        self.convblock6 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 6, RF = 22

        self.convblock7 = self.create_conv2d(10, 16, dropout=dropout_value, padding=0, bn=False, relu=False) # output_size = 4, RF = 26

        # OUTPUT BLOCK
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 

        self.convblock8 = self.create_conv2d(16, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # output_size = 1, RF = 26
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_planes, planes, stride=1 ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    






class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.trainer = None
        self.name = "Models"

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
        
    def summary(self, input_size):
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None


def ResNet18(num_class=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_class)


#implementation of the new resnet model
class newResnetS11(Net):
  def __init__(self,name="Model",dropout_value=0):
    super(newResnetS11,self).__init__(name)
    self.prepLayer=self.create_conv2d(3, 64, dropout=dropout_value)
    #layer1
    self.layer1Conv1=self.create_conv2d(64,128, dropout=dropout_value,max_pooling=1)
    self.layer1resnetBlock1=self.resnetBlock(128,128)
    #layer2
    self.layer2Conv1=self.create_conv2d(128,256, dropout=dropout_value,max_pooling=1)
    #layer3
    self.layer3Conv1=self.create_conv2d(256,512, dropout=dropout_value,max_pooling=1)
    self.layer3resnetBlock1=self.resnetBlock(512,512)
    #ending layer or layer-4
    self.maxpool=nn.MaxPool2d(4,1)
    self.fc_layer=self.create_conv2d(512, 10, kernel_size=(1,1), padding=0, bn=False, relu=False)
  def resnetBlock(self,in_channels, out_channels):
      l=[]
      l.append(nn.Conv2d(in_channels,out_channels,(3,3),padding=1,bias=False))
      l.append(nn.BatchNorm2d(out_channels))
      l.append(nn.ReLU())
      l.append(nn.Conv2d(in_channels,out_channels,(3,3),padding=1,bias=False))
      l.append(nn.BatchNorm2d(out_channels))
      l.append(nn.ReLU())
      return nn.Sequential(*l)

  def forward(self,x):
    #prepLayer
    x=self.prepLayer(x)
    #Layer1
    x=self.layer1Conv1(x)
    r1=self.layer1resnetBlock1(x)
    x=torch.add(x,r1)
    #layer2
    x=self.layer2Conv1(x)
    #layer3
    x=self.layer3Conv1(x)
    r2=self.layer3resnetBlock1(x)
    x=torch.add(x,r2)
    #layer4 or ending layer
    x=self.maxpool(x)
    x=self.fc_layer(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)


class depth_model(Net):
  def __init__(self,name="Model",dropout_value=0.0):
    super(depth_model,self).__init__(name)
    #first part of architecture of solving the mask image
    self.conv1=self.create_conv2d(3,32)
    self.conv2=self.create_conv2d(32,64)
    self.conv3=self.create_conv2d(64,128)
    self.conv4=self.create_conv2d(128,128)
    self.conv5=self.create_conv2d(128,1,(1,1),padding=0,relu=False,bn=False)
    #second part of architecture for solving the depth image 
    self.conv7=self.create_conv2d(3,32)
    self.conv8=self.create_conv2d(32,64)
    self.conv9=self.create_conv2d(64,128)
    self.conv10=self.create_conv2d(128,128)
    self.conv11=self.create_conv2d(256,256)
    self.conv12=self.create_conv2d(256,1,(1,1),padding=0,relu=False,bn=False)

  def forward(self,x1,x2):
    #first part for finding mask
    x1=self.conv1(x1)
    x1=self.conv2(x1)
    x1=self.conv3(x1)
    x1=self.conv4(x1)
    self.value=x1
    x1=self.conv5(x1) #first output for mask
    #second part for finding depth 
    x2=self.conv7(x2)
    x2=self.conv8(x2)
    x2=self.conv9(x2)    
    x2=self.conv10(x2)
    #x2=self.upsample(x2)
    #print(x.shape)
    x2=torch.cat((x2,self.value),1)
    #print(x.shape)
    x2=self.conv11(x2)
    x2=self.conv12(x2)
    return x1,x2



