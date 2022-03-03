import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse

def gpClassifier(model_c, sgp, tgp):
    #gpdc = gpDomainClassifier(sgp.shape[1] * sgp.shape[0], sgp.shape[0]).cuda()
    #type(sgp.cuda())
    #pdb.set_trace()
    sgp_scores = model_c(grad_reverse(sgp.view(-1))).view(sgp.shape[0],1)
    tgp_scores = model_c(grad_reverse(tgp.view(-1))).view(sgp.shape[0],1)
    #pdb.set_trace()    
    sgp_labels = Variable(torch.ones_like(sgp_scores).float().cuda())
    loss_sgp_domain = F.binary_cross_entropy(sgp_scores, sgp_labels)
    tgp_labels = Variable(torch.zeros_like(tgp_scores).float().cuda())
    loss_tgp_domain = F.binary_cross_entropy(tgp_scores, tgp_labels)
    #pdb.set_trace()
    loss_gp_domain = (loss_sgp_domain + loss_tgp_domain) / 2
    return loss_gp_domain
    


class gpDomainClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(gpDomainClassifier, self).__init__()
        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(self.fc3(x))
        # x = x.view(x.size(0), -1)
        return x

