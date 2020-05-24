import torch.optim as optim
from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import time
class lrRangeFinder():
  def __init__(self,model,dataloader,criterion,optimizer):
    self.model=model
    self.dataloader=dataloader
    self.learning_rates=[]
    self.training_accuracy=[]
    self.loss_list=[]
    self.learning_rate=0.00001
    self.average_accuracy=0.0
    self.average_loss=0.0
    self.criterion=criterion
    self.optimizer=optimizer
    self.scheduler=False
    self.use_amp=False
  def plot(self,epochs):
    for i in range(epochs):
        self.model.train()
        torch.backends.cudnn.benchmark = True
        pbar = tqdm_notebook(self.dataloader)
        self.optimizer.param_groups[0]['lr']=self.learning_rate
        for data1,data2,target1,target2 in pbar:
            # get
            #start1=time.time() 
            #start=time.time()
            data1,data2,target1,target2 = data1.to(self.model.device),data2.to(self.model.device),target1.to(self.model.device), target2.to(self.model.device)
            #print("loading data into cuda time is {}".format(time.time()-start))
            # Init
            #start=time.time()
            optimizer.zero_grad()
            #print("loading data optimizer zero grad time is {}".format(time.time()-start))
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            #start=time.time()
            output1,output2 = self.model(data1,data2)
            #print("loading data into model and getting output time is {}".format(time.time()-start))
            #start=time.time()
            output1,output2=output1.squeeze(1),output2.squeeze(1)
            self.loss1=self.criterion[0](output1,target1+0.00000001)
            self.loss2=self.criterion[1](output2,target2)
            self.loss=(self.loss1+self.loss2)
            #print("calculating loss and getting output time is {}".format(time.time()-start))
            #Implementing L1 regularization
            #start=time.time()      
            if self.use_amp:
              with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
                  scaled_loss.backward()
            else:
              self.loss.backward()
            #print("loss backward into  time is {}".format(time.time()-start))
            
            #start=time.time()
            optimizer.step()
            #print("optimizer step is {}".format(time.time()-start))
            #start=time.time()
            # Update pbar-tqdm
            correct1 = output1.long().eq(target1.long().view_as(output1.long())).float().mean().item()
            correct2 = output2.long().eq(target2.long().view_as(output2.long())).float().mean().item()

            correct=(correct1+correct2)/2.0
            self.average_accuracy+=correct
            self.average_loss+=self.loss
            #print("completiion of accuracy calculation time is {}".format(time.time()-start))
            #print("the learning rate is {}".format(optimizer.param_groups[0]['lr']))
            #print("completiion of entire batch time is {}".format(time.time()-start1))
        self.learning_rates.append(self.learning_rate)
        self.loss_list.append(self.average_loss/len(self.dataloader))
        self.training_accuracy.append(self.average_accuracy/len(self.dataloader))
        self.learning_rate*=10
    return self.learning_rates,self.training_accuracy,self.loss_list        

