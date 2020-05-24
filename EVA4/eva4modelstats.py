import json
import torch
#TODO: pass save format also, can be pickle or json
class ModelStats:
  def __init__(self, model, path):
    self.model = model
    self.path = path
    self.batch_train_loss = []
    self.batch_train_acc = []
    self.batch_lr = []
    
    self.avg_test_loss = []
    self.test_acc = []
 
    self.train_acc = []
    self.avg_train_loss = []
    self.lr = []
 
    self.batches = 0
    self.epochs = 0
 
    self.curr_train_acc = 0
    self.curr_train_loss = 0
    self.curr_test_acc = 0
    self.curr_test_loss = 0
    self.train_samples_seen = 0
    self.test_samples_seen = 0
    self.best_test_loss = 100000
    self.misclassified_images = []
    
 
  def add_batch_train_stats(self, loss, acc, cnt, lr):
    self.batches += 1
    self.batch_train_loss.append(loss)
    self.batch_train_acc.append(acc)
    self.curr_train_loss += loss
    self.curr_train_acc = acc
    self.train_samples_seen += cnt
    self.batch_lr.append(lr)
    
 
  def add_batch_test_stats(self, loss, acc, cnt):
    self.curr_test_loss += loss
    self.curr_test_acc = acc
    self.test_samples_seen += cnt
 
  def next_epoch(self, lr):
    self.epochs += 1
    #print(self.curr_test_loss, self.test_samples_seen, self.curr_train_loss, self.train_samples_seen)
    self.avg_test_loss.append(self.curr_test_loss/self.test_samples_seen)
    self.test_acc.append(self.curr_test_acc)
    self.avg_train_loss.append(self.curr_train_loss/self.train_samples_seen)
    self.train_acc.append(self.curr_train_acc)
    self.lr.append(lr)
    self.curr_train_acc = 0
    self.curr_train_loss = 0
    self.curr_test_acc = 0
    self.curr_test_loss = 0
    self.train_samples_seen = 0
    self.test_samples_seen = 0
 
    if self.epochs == 1 or self.best_test_loss > self.avg_test_loss[-1]:
      print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.best_test_loss,self.avg_test_loss[-1]))
      torch.save(self.model.state_dict(), f"{self.path}/{self.model.name}.pt")
      self.best_test_loss = self.avg_test_loss[-1]
 
  def save(self):
    pass
    '''s = {"batch_train_loss":self.batch_train_loss.cpu().numpy() if(isinstance(self.batch_train_loss, torch.Tensor)) else self.batch_train_loss, "batch_train_acc":self.batch_train_acc.cpu().numpy(),
         if(isinstance(self.batch_train_acc, torch.Tensor)) else self.batch_train_loss,"batch_lr":self.batch_lr.cpu().numpy(), "avg_test_loss": self.avg_test_loss.cpu().numpy(), "test_acc": self.test_acc.cpu().numpy(),
         "train_acc": self.train_acc.cpu().numpy(), "avg_train_loss" : self.avg_train_loss.cpu().numpy(), "lr": self.lr.cpu().numpy(),
         "best_test_loss": self.best_test_loss.cpu().numpy(), "epochs": self.epochs.cpu().numpy()}
    with open(f'{self.path}/{self.model.name}_stats.json', 'w') as fp:
      json.dump(s, fp, sort_keys=True, indent=4)'''
 
 
  def get_latest_batch_desc(self):
    if len(self.batch_train_loss)==0:
      return "first batch"
    return f'Batch={self.batches} Loss={self.batch_train_loss[-1]:0.4f} Acc={100*self.curr_train_acc:0.2f}%'
  
  def get_misclassified_images(self):
    return self.misclassified_images
  
  def get_epoch_desc(self):
    return f'Epoch: {self.epochs}, Train set: Average loss: {self.avg_train_loss[-1]:.4f}, Accuracy: {100*self.train_acc[-1]:.2f}%; Test set: Average loss: {self.avg_test_loss[-1]:.4f}, Accuracy: {100*self.test_acc[-1]:.2f}%'