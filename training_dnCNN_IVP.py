#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The first step is loading our pickel files for dataset


# In[2]:


import pickle
import numpy as np
infile = open("train.pkl",'rb')
dataset = pickle.load(infile) #x,y or clean, noisety
infile.close()


# In[3]:


#dataset[0]=first x y pair
#dataset[1]=second x y pair


# In[4]:


import torch
train_loader = torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True)


# In[5]:


#Now that the dataset is prepared and shuffled, we can call the model function


# In[6]:


#get_ipython().system('ls')


# In[7]:


#pip install import-ipynb
#https://stackoverflow.com/questions/20186344/importing-an-ipynb-file-from-another-ipynb-file


# In[8]:


import import_ipynb


# In[9]:


import DnCNN_IVP


# In[33]:


#Now that it is imported, let us create an instance
# model=DnCNN_IVP.DnCNN(1,1,20)
device=torch.device('cuda:0')
# model.to(device)


# In[11]:


get_ipython().system('pip install torchsummary')


# In[12]:


from torchsummary import summary
# summary(model,input_size=(1,180,180))


# In[13]:


N=200
from torch import nn
class loss_new(nn.Module): #N is dataset size
  def __init__(self):
    super(loss_new,self).__init__()
  def forward(self,out,y,x):
    return torch.norm(out-(y-x))/(2*N) #Where does N get defined?


# In[14]:


import torch
from torch import optim
# optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0001)


# In[15]:


criterion=loss_new()


# In[20]:


get_ipython().system('pip install tqdm')


# In[51]:


#Need to set a batch size of 128
#Tensor to data loader
def training_function(net, loader, optim, scheduler, model_name, sigma, epochs=50):
  from tqdm import tqdm
  loss_list=[]
  net.train()
  """
    y is the noisy image and it is used as input

    x is the noiseless image and it is used as a target label
  """
    
  for e in tqdm(range(epochs)): #For each epoch
    train_loss=0
    for i, clean in enumerate(loader):
      optim.zero_grad()#zero grad so they don't stack
      clean = clean.to(device, dtype=torch.float)
      noisy = clean + sigma * torch.from_numpy(np.random.randn(40,40)).to(device, dtype=torch.float)
      clean = clean[:,None,:,:]
      noisy = noisy[:,None,:,:].to(device, dtype=torch.float)
      prediction = net(noisy)
      batch_loss = criterion(prediction, noisy, clean)
      batch_loss.backward()
      optim.step()
      scheduler.step()
      train_loss += batch_loss.item()
      if i % 5000 == 0:
          print('training at: ', i)
    train_loss = train_loss/len(loader)
    loss_list.append(train_loss)

    print ("Epoch {}: Has a loss of Loss: {:.3f}".format(e+1 ,train_loss))

  torch.save(model.state_dict(), model_name + '.pt')
  return loss_list
  #return np.mean(train_loss)


# In[ ]:


from torch.optim.lr_scheduler import StepLR
model=DnCNN_IVP.DnCNN(1,1,20)
device=torch.device('cuda:0')
model.to(device)
epoch = 50

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.001**(1/50))
print('beginning with training 15')

plot_me_15=training_function(model, train_loader, optimizer, scheduler, 'dncnn15', 15, 50)


# In[ ]:


plot_me_25=training_function(model, train_loader, optimizer, scheduler, 'dncnn25', 25, 50)


# In[ ]:


plot_me_50=training_function(model, train_loader, optimizer, scheduler, 'dncnn50', 50, 50)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot([i for i in range(1,51)], plot_me_15, 'red', label= "noise-15")
plt.plot([i for i in range(1,51)], plot_me_25, 'black', label= "noise-25")
plt.plot([i for i in range(1,51)], plot_me_50, 'blue', label= "noise-50")
plt.legend()


# In[ ]:


# We can't use a learning rate as high as theirs because we didn't take patches so we have less data and if we just learn super fast, model will explode and oscilate


# In[ ]:


# torch.save(model.state_dict(), "dncnn15_0point00001.pt")


# In[ ]:


# loaded=torch.load("dncnn15_0point0001.pt")


# In[ ]:


#6 zeros slow
#3 zeros great
#1 zero explodes


# In[ ]:


# model=DnCNN_IVP.DnCNN(1,1,20)
# model.to(device)


# In[ ]:


# criterion=loss_new()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)


# In[ ]:


# infile = open("BSDS_25.pkl",'rb')
# dataset = pickle.load(infile) #x,y or clean, noisety
# infile.close()
# train_loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)


# In[ ]:


# plot_me=training_function(model,train_loader,optimizer,50)


# In[ ]:




