import argparse
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models 
from diagnosis import mirt
import torchvision
import torchmetrics
from torchmetrics.classification import MulticlassCalibrationError
import torch.nn.functional as F

class AbilityWeightedEnsemble(nn.Module):
    def __init__(self, models, ability_dim,feature_dim, device='cuda:0'):
        super(AbilityWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.mlp = nn.Sequential(
            nn.Linear(ability_dim*self.num_models+feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_models),
            nn.Softmax(dim=1)
        )
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Linear(2048, feature_dim)

        self.device = device
        for model in self.models:
            model.to(device)
        self.mlp.to(device)
        self.feature_extractor.to(device)

    def forward(self, x, cdm):
        outputs = []
        ability_vector = cdm.irt_net.theta.weight.data.detach().to(self.device).reshape(-1)
        ability_vector = torch.vstack([ability_vector for i in range(x.shape[0])])
        batch_size = x.shape[0]
        sample_feature = self.feature_extractor(x).reshape(batch_size,-1)
        mlp_input = torch.cat((sample_feature, ability_vector), dim=1)
        weight = self.mlp(mlp_input).squeeze().unsqueeze(2)
        for i,model in enumerate(self.models):
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=-1)
        weighted_outputs = torch.bmm(outputs, weight).squeeze(2)
        return weighted_outputs
    
    def train(self):
        for model in self.models:
            model.train()
      
    def eval(self):
        for model in self.models:
            model.eval()

    def get_ability(self,cdm):
        print("model ability:")
        print(cdm.irt_net.theta.weight.data.detach().cpu().numpy())
        logging.info("model ability:")
        logging.info(cdm.irt_net.theta.weight.data.detach().cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16','vgg19','vgg11','densenet121','mobilenet'])
    parser.add_argument('--num_models', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=35)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--rand_fraction', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--mirt_dim',type=int ,default=20)
    parser.add_argument('--feature_dim',type=int ,default=20)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()

    # Set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p', filename='log/made'+args.dataset+'_'+args.model+'_seed=' + str(args.seed) + '.log')
    logger = logging.getLogger(__name__)

    # Set up device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # Set random seed
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    

    # Set up data loader
    if args.dataset == 'cifar10':
        from dataset.cifar10 import get_CIFAR10_train_valid_split_loader, get_CIFAR10_test_loader
        train_loader, valid_loader = get_CIFAR10_train_valid_split_loader(args.batch_size, args.rand_fraction, args.num_workers, valid_size=args.valid_size)
        test_loader = get_CIFAR10_test_loader(args.batch_size, args.num_workers)
    elif args.dataset == 'cifar100':
        from dataset.cifar100 import get_CIFAR100_train_valid_split_loader, get_CIFAR100_test_loader
        train_loader, valid_loader = get_CIFAR100_train_valid_split_loader(args.batch_size, args.rand_fraction, args.num_workers, valid_size=args.valid_size)
        test_loader = get_CIFAR100_test_loader(args.batch_size, args.num_workers)
    elif args.dataset == 'svhn':
        from dataset.svhn import get_SVHN_train_valid_split_loader, get_SVHN_test_loader
        train_loader, valid_loader = get_SVHN_train_valid_split_loader(args.batch_size, args.rand_fraction, args.num_workers, valid_size=args.valid_size)
        test_loader = get_SVHN_test_loader(args.batch_size, args.num_workers)
    else:
        raise NotImplementedError
    
    # Set up model
    model_list = []
    for i in range(args.num_models):
        model = models.get_network(args, args.model)
        model_list.append(model)
    ensemble_model = AbilityWeightedEnsemble(model_list, args.mirt_dim,args.feature_dim, device)
    cdm = mirt.MIRT(args.num_models, len(valid_loader.dataset), args.mirt_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ensemble_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_classes = 100 if args.dataset == 'cifar100' else 10
    mece = MulticlassCalibrationError(num_classes=num_classes, n_bins=num_classes, norm='l1')
    best_acc = 0.0
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            ensemble_model.train()
            images = images.to(device)
            labels = labels.to(device)
            outputs = ensemble_model(images,cdm)
            loss = criterion(outputs, labels)
            for model in (ensemble_model.models):
                loss += criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % args.log_interval == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_epochs, i+1, total_step, loss.item()))
               
        ensemble_model.eval()
        base_model_outputs = []
        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            for j,model in enumerate(ensemble_model.models):
                _,predicted = torch.max(model(images).data, 1)
                for k in range(len(predicted)):
                    base_model_outputs.append((j, k+args.batch_size*i, 1 if predicted[k]==labels[k] else 0))

        model_tensor = torch.tensor([out[0] for out in base_model_outputs],dtype=torch.int64)
        item_tensor = torch.tensor([out[1] for out in base_model_outputs], dtype=torch.int64)
        response_tensor = torch.tensor([out[2] for out in base_model_outputs], dtype=torch.float32)
        cdm_dataset = torch.utils.data.TensorDataset(model_tensor, item_tensor, response_tensor)
        cdm_loader =  torch.utils.data.DataLoader(cdm_dataset, batch_size=128, shuffle=True)
        cdm.train(cdm_loader, epoch=15, device=device)
        ensemble_model.get_ability(cdm)
        correct = 0
        total = 0   
        nll = 0
        ensemble_model.eval()
        for images,labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)     
            outputs = ensemble_model(images,cdm)
            nll += F.cross_entropy(outputs, labels, reduction='sum').item()
            numbers,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            mece.update(outputs, labels)
        ece = mece.compute()
        print("ece:%.5f" % ece)
        logger.info("ece:%.5f" % ece)
        mece.reset()
        print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
        logger.info('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
        print('Testing NLL : %.5f' % ( nll / total))
        logger.info('Testing NLL : %.5f' % ( nll / total))
        if (correct / total) > best_acc:
            best_acc = correct / total
            torch.save(ensemble_model.state_dict(), os.path.join(args.save_dir, args.dataset+'_'+str(args.num_models)+'*'+args.model+'_seed=' + str(args.seed) + 'epoch=' + str(epoch+1) + 'made.ckpt'))         
            torch.save(cdm.irt_net.state_dict(), os.path.join(args.save_dir, args.dataset+'_'+str(args.num_models)+'*'+args.model+'_seed=' + str(args.seed) + 'epoch=' + str(epoch+1) + 'mirt.ckpt'))

    # Test the model
    ensemble_model.eval()
    correct = 0
    total = 0
    nll = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = ensemble_model(images,cdm)
        nll += F.cross_entropy(outputs, labels, reduction='sum').item()
        numbers,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        mece.update(outputs, labels)
    ece = mece.compute()
    print("ece:%.5f" % ece)
    logger.info("ece:%.5f" % ece)
    mece.reset()
    print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
    logger.info('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
    print('Testing NLL : %.5f' % ( nll / total))
    logger.info('Testing NLL : %.5f' % ( nll / total))


        