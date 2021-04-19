import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model = MyModel()
model.fc3.register_forward_hook(get_activation('fc3'))
model.cl2.register_forward_hook(get_activation('cl2'))
x = torch.randn(0,2)
time_file_name = 'H:\\paper\\P-idea-1\\test_2\\time\\x.pkl'
with open(time_file_name, 'wb') as f:
    #x = pickle.load(f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
    pickle.dump(x, f)
# with open(time_file_name, 'rb') as f:
#     [x] = pickle.load(f)
output = model(x)


# with open(file_name, 'wb') as f:
#     pickle.dump([acc_list, loss_list], f)
        # pickle.dump([acc_list_1, loss_list_1], f)
time_file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\time\\1.pkl'
time_file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\time\\2.pkl'
with open(time_file_name_1, 'wb') as f1:
    pickle.dump([activation['cl2']], f1)
    pickle.dump([activation['cl2']], f1)
with open(time_file_name_2, 'wb') as f2:
    pickle.dump([activation['fc3']], f2)
print(activation['cl2'])
print(activation['fc3'])