import os
import paddle
import numpy as np
from paddle import optimizer,nn
from model import EncodeProcessDecode
from pgl.utils.data.dataloader import Dataloader, Dataset
from log import logger_setup
from utils import log_images
from cylinder_data import MeshcylinderDataset
from data import MeshAirfoilDataset
from pathlib import Path 

def collate_fn(batch_data):
    return batch_data

Airfoil_ImagePath = './result/image/airfoil/'
Cylinder_ImagePath = './result/image/cylinder/'
if (os.path.exists(Airfoil_ImagePath)) == False:
    os.makedirs(Airfoil_ImagePath)
if (os.path.exists(Cylinder_ImagePath)) == False:
    os.makedirs(Cylinder_ImagePath)

class Trainer:
    def __init__(self, my_type, mode):
        if my_type == "cylinder":
            self.data = MeshcylinderDataset('./data/cylinderdata/', mode)
        elif my_type == "airfoil":
            self.data = MeshAirfoilDataset('./data/NACA0012_interpolate/', mode)
        self.loader = self.dataloader(mode)

        if my_type == "cylinder":
            self.log_path = Path('./result/log/cylinder.log')
        elif my_type == "airfoil":
            self.log_path = Path('./result/log/airfoil.log')

        if my_type == "cylinder":
            self.model_path = './result/modelcylinder.pkl'
        elif my_type == "airfoil":
            self.model_path = './result/modelairfoil.pkl'
        
        self.model = EncodeProcessDecode(output_size=3,
                                latent_size=128,
                                num_layers=2,
                                message_passing_aggregator='sum', message_passing_steps=6, mode = my_type)

    def dataloader(self, mode):
        dataset = []
        for i in range(self.data.len):
            Data = self.data.get(i)
            dataset.append(Data)
        if mode == "train":
            batch_size = 4
        elif mode == "test":
            batch_size = 1
        loader = Dataloader(dataset, batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
        return loader
           
paddle.device.set_device("gpu:0")

#####################################################Set model parameters，airfiol or cylinder
# my_type = "airfoil"
my_type = "cylinder"
###############################################
#  for train
###############################################
mode = "train"
trainer = Trainer(my_type, mode)
loader = trainer.loader
model = trainer.model

optimizers = optimizer.Adam(parameters=model.parameters(), learning_rate=0.0005)
scheduler = optimizer.lr.ExponentialDecay(0.0005, 0.1 + 1e-6, last_epoch=-1)
criterion =nn.MSELoss()
if (os.path.exists('./result/log/')) == False:
    os.makedirs('./result/log/')
root_logger = logger_setup(trainer.log_path)
model.train()
root_logger.info("===========start train===========")

for epoch in range(500):
    sum_loss=0
    root_logger.info("Epoch"+str(epoch+1)) 
    for batch in loader:
        truefield = []
        for graph in batch:
            truefield.append(graph.y)
        truefield = paddle.concat(truefield, axis=0)
        prefield = model(batch)
        mes_loss = criterion(prefield,truefield)
        optimizers.clear_grad()
        mes_loss.backward()
        optimizers.step()
        sum_loss += mes_loss.item()
    print('epoch=', epoch)
    print(sum_loss)
    avg_loss=(sum_loss)/len(loader)
    root_logger.info("        trajectory_loss")
    root_logger.info("        " + str(avg_loss))  
    if((epoch==60)|(epoch==100)|(epoch==140)|(epoch==170)): 
        scheduler.step() 
paddle.save(model.state_dict(), trainer.model_path) 

####################################################
# for test
####################################################
mode = "test"
trainer = Trainer(my_type, mode)
loader = trainer.loader
model = trainer.model 
model.load_dict(paddle.load(trainer.model_path))
root_logger.info("===========start test===========") 

model.eval()
with paddle.no_grad():
    sum_loss=0
    for index, batch in enumerate(loader):
        truefield=batch[0].y
        prefield=model(batch)
        log_images(batch[0].pos, prefield,truefield,trainer.data.elems_list, 'test', index, flag = my_type)
        mes_loss=criterion(prefield,truefield)
        sum_loss+=mes_loss.item()
avg_loss = sum_loss/(len(loader))
root_logger.info("        trajectory_loss")
root_logger.info("        " + str(avg_loss))