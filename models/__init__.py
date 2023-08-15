import torchvision.models as models
import torch.nn as nn


def get_network(args,model_name):
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    elif args.dataset == 'svhn':
        num_classes = 10
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        raise NotImplementedError
    if model_name == 'resnet18':
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        from torchvision.models import ResNet34_Weights
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet101':
        from torchvision.models import ResNet101_Weights
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet152':
        from torchvision.models import ResNet152_Weights
        model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg11':
        from torchvision.models import VGG11_Weights
        model = models.vgg11(weights=VGG11_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg16':
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg19':
        from torchvision.models import VGG19_Weights
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg11_bn':
        from torchvision.models import VGG11_BN_Weights
        model = models.vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg16_bn':
        from torchvision.models import VGG16_BN_Weights
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg19_bn':
        from torchvision.models import VGG19_BN_Weights
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'densenet121':
        from torchvision.models import DenseNet121_Weights
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet':
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise NotImplementedError
    # model = model.to(args.device)
    return model