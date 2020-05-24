from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from eva4modelstats import ModelStats
# https://github.com/tqdm/tqdm
class Train:
  def __init__(self, model, dataloader, optimizer, stats, scheduler=None, L1lambda = 0,criterion=None):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda
    self.criterion=criterion
    self.loss=0.0
    self.loss1=0.0
    self.loss2=0.0
    self.use_amp=False
  def run(self):
    self.model.train()
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    pbar = tqdm_notebook(self.dataloader)
    for data1,data2,target1,target2 in pbar:
      # get 
      data1,data2,target1,target2 = data1.to(self.model.device),data2.to(self.model.device),target1.to(self.model.device), target2.to(self.model.device)
 
      # Init
      self.optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
 
      # Predict
      output1,output2 = self.model(data1,data2)
      output1,output2=output1.squeeze(1),output2.squeeze(1)
      #print("depth max is {}".format(output2.max()))
      #print("mask max is {}".format(output1.max()))
      #print(target2.max())

      self.loss1=self.criterion[0](output1,target1)
      self.loss2=self.criterion[1](output2,target2)
      self.loss=(self.loss1+self.loss2)
      #print(self.loss1)
      #print(self.loss2)
      #Implementing L1 regularization      
      if self.use_amp:
        with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
      else:
        self.loss.backward()
      self.optimizer.step()
 
      # Update pbar-tqdm
      correct1 = output1.long().eq(target1.long().view_as(output1.long())).float().mean().item()
      correct2 = output2.long().eq(target2.long().view_as(output2.long())).float().mean().item()
      correct=(correct1+correct2)/2.0
      #print(correct)
      lr = 0
      if self.scheduler:
        lr = self.scheduler.get_last_lr()[0]
      else:
        # not recalling why i used sekf.optimizer.lr_scheduler.get_last_lr[0]
        lr = self.optimizer.param_groups[0]['lr']
      
      #lr =  if self.scheduler else (self.optimizer.lr_scheduler.get_last_lr()[0] if self.optimizer.lr_scheduler else self.optimizer.param_groups[0]['lr'])
      
      self.stats.add_batch_train_stats(self.loss.item(), correct, len(data1), lr)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if self.scheduler:
        self.scheduler.step()
 
class Test:
  def __init__(self, model, dataloader, stats,writer, scheduler=None,criterion=None):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.scheduler = scheduler
    self.loss=0.0
    self.loss1=0.0
    self.loss2=0.0
    self.average_loss=0.0
    self.criterion=criterion
    self.writer=writer
  def run(self):
    self.model.eval()
    with torch.no_grad():
        for data1,data2,target1,target2 in self.dataloader:
            data1,data2,target1,target2 = data1.to(self.model.device),data2.to(self.model.device),target1.to(self.model.device), target2.to(self.model.device)
            output1,output2 = self.model(data1,data2)
            output1,output2=output1.squeeze(1),output2.squeeze(1)
            self.loss1=self.criterion[0](output1,target1)
            self.loss2=self.criterion[1](output2,target2)
            self.loss=(self.loss1+self.loss2)  # sum up batch loss
            self.average_loss+=self.loss
            correct1 = output1.long().eq(target1.long().view_as(output1.long())).float().mean().item()
            correct2 = output2.long().eq(target2.long().view_as(output2.long())).float().mean().item()
            correct=(correct1+correct2)/2.0
            #print(correct)
            self.stats.add_batch_test_stats(self.loss, correct, len(data1))
        self.average_loss/=len(self.dataloader)
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
              #we are trying to step the scheduler using average loss
              self.scheduler.step(self.average_loss)
 
class Misclass:
  def __init__(self, model, dataloader, stats,criterion=None):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.criterion=criterion
    self.loss=0.0
  def run(self):
    self.model.eval()
    with torch.no_grad():
        for data, target in self.dataloader:
          if len(self.stats.misclassified_images) == 25:
            return
          data, target = data[0].to(self.model.device), target.to(self.model.device)
          target=target.view(-1)
          output = self.model(data)
          if(self.criterion):
             self.loss=self.criterion(output, target)
          else:
             self.loss = F.nll_loss(y_pred, target) # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          is_correct = pred.eq(target.view_as(pred))
          misclassified_inds = (is_correct==0).nonzero()[:,0]
          for mis_ind in misclassified_inds:
            if len(self.stats.misclassified_images) == 25:
               break
            self.stats.misclassified_images.append({"target": target[mis_ind].cpu().numpy(), "pred": pred[mis_ind][0].cpu().numpy(),"img": data[mis_ind]})
def return_image(img):
  img=img.cpu().numpy()
  for i,row in enumerate(img):
    for j,value in enumerate(row):
      if(value>0):
        img[i,j]=255
      else:
        img[i,j]=0
  return img
 
 
def printing_results(model,dataloader,epoch):
   epoch+=8
   with torch.no_grad():
            data1,data2,target1,target2=iter(dataloader).next()
            data1,data2,target1,target2 = data1.to(model.device),data2.to(model.device),target1.to(model.device), target2.to(model.device)
            output1,output2 = model(data1,data2)
            output1,output2=output1.squeeze(1),output2.squeeze(1) 
            figure=plt.figure(figsize=(10,10))
            row=4
            col=4
            k=0
            print("masks")
            for i in range(1,row*col+1,2):
              figure.add_subplot(row,col,i)
              plt.imshow(target1[i-1].cpu().numpy()*255,cmap="gray")
              figure.add_subplot(row,col,i+1)
              plt.imshow((output1[i-1]).cpu().numpy()*255,cmap="gray")
              k+=1
            plt.savefig('/content/drive/My Drive/results_images/mask{}.jpg'.format(epoch))
            plt.show()
            figure=plt.figure(figsize=(10,10))
            row=4
            col=4
            k=0
            print("scaling masks between 0 and 1")
            for i in range(1,row*col+1,2):
              figure.add_subplot(row,col,i)
              plt.imshow(target1[i-1].cpu().numpy()*255,cmap="gray")
              figure.add_subplot(row,col,i+1)
              plt.imshow(return_image(output1[i-1])*255,cmap="gray")
              k+=1
            plt.savefig('/content/drive/My Drive/results_images/mask_scale{}.jpg'.format(epoch))
            plt.show()          
            print("printing depth images")       
            figure=plt.figure(figsize=(10,10))
            row=4
            col=4
            for i in range(1,row*col+1,2):
              figure.add_subplot(row,col,i)
              plt.imshow(target2[i-1].cpu().numpy()*255,cmap="gray")
              figure.add_subplot(row,col,i+1)
              plt.imshow(output2[i-1].cpu().numpy()*255,cmap="gray")
            plt.savefig('/content/drive/My Drive/results_images/depth{}.jpg'.format(epoch))
            plt.show()
class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, statspath,criterion,writer,scheduler=None, batch_scheduler=False, L1lambda = 0):
    self.model = model
    self.scheduler = scheduler
    self.criterion=criterion
    self.batch_scheduler = batch_scheduler
    self.optimizer = optimizer
    self.stats = ModelStats(model, statspath)
    self.train = Train(model, train_loader, optimizer, self.stats, self.scheduler if self.batch_scheduler else None, L1lambda,criterion)
    self.test = Test(model, test_loader, self.stats,writer,self.scheduler,criterion)
    self.misclass = Misclass(model, test_loader, self.stats)
    self.test_loader=test_loader
    torch.backends.cudnn.benchmark = True
    
  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      self.train.run()
      self.test.run()
      lr = self.optimizer.param_groups[0]['lr']
      self.stats.next_epoch(lr)
      pbar.write(self.stats.get_epoch_desc())
      # need to ake it more readable and allow for other schedulers
      if self.scheduler and not self.batch_scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.scheduler.step()
      pbar.write(f"Learning Rate = {lr:0.6f}")
      print("printing results")
      printing_results(self.model,self.test_loader,epoch)
 
    # save stats for later lookup
    self.stats.save()