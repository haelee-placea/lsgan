import os
import torch
import pickle


# def save(save_dir, model_name, G, D,):
#     save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
#     torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

#     with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
#         pickle.dump(self.train_hist, f)