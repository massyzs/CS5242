import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset import ImageDataset, DomainDataset
from torch.utils.data import DataLoader,random_split
from dann import LeNetFeatureExtractor, LabelPredictor, DomainDiscriminator, DANN
from itertools import zip_longest
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter


# config={
#     "mode": "train",
#     "batch": 64,
#     "epoch": 20,
#     "lr": 1e-4,
#     "cuda": 0,
#     "norm": 1,
#     "norm_type": "BN",
#     "dropout": False,
#     "weight_decay": False,
#     "opt": "adam",
#     "activation": "leaky_relu",
#     "data_augmentation":False,
#     "save":False,
#     "dropout_rate":0.8,
#     "alpha": 0.1,  # You can set your desired initial alpha value here.
#     "domain_weight": 0.7,
#     "source_domain_label": 0,
# }

parser = argparse.ArgumentParser(description='NA')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--norm', type=int, default=1, help="which layer to add normalization: 1-8")
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--dropout', type=int, default=1, help="bool: whether or not to use drop out")
parser.add_argument('--weight_decay', type=int, default=0, help="bool: whether or not to use weight decay (L2 regularization)")
parser.add_argument('--norm_type', type=str, default="BN",help="BN for batchnorm, LN for LayerNorm")
parser.add_argument('--opt', type=str, default="adam", help="optimizer type: adam or sgd")
parser.add_argument('--activation', type=str, default="relu", help="leakyrelu, relu, sigmoid, tanh")
parser.add_argument('--aug', type=int, default=1, help="bool: whether or not to use data augmentation")
# parser.add_argument('--save', type=int, default=0, help="bool: whether or not to use data augmentation")
parser.add_argument('--rate', type=float, default=0.7, help="drop out rate")
parser.add_argument('--ratio', type=float, default=0.5, help="ratio of test set used in train set")
args = parser.parse_args()

writer = SummaryWriter(f'tfboard/DANN_testSplit{args.ratio}_dropout{args.rate}')

config={
    "mode": "train",
    "batch": 64,
    "epoch": 40,
    "lr": 1e-4,
    "cuda": args.cuda,
    "norm": 1,
    "norm_type": "BN",
    "dropout": True,
    "weight_decay": False,
    "opt": "adam",
    "activation": "relu",
    "data_augmentation":False,
    "dropout_rate":args.rate,
    "ratio":args.ratio,
}

base_dir="dataset/"
device=config["cuda"]
device=torch.device(f"cuda:{device}")

# Define the hyperparameters
# batch_size = 128
# lr = 0.001
# num_epochs = 10
batch_size = config["batch"]
lr = config["lr"]
num_epochs = config["epoch"]
lambda_val = 0.1  # domain adversarial loss weight

# Define the datasets and data loaders
trainset = DomainDataset(base_dir + config["mode"], device=device, config=config, domain=0, train=True)
test_dataset = DomainDataset(base_dir + 'test', device=device, config=config, domain=1, train=False)
target_domain_dataset, testdataset = random_split(test_dataset, [int(0.1 * len(test_dataset)), len(test_dataset)-int(0.1 * len(test_dataset))])
trainloader = DataLoader(torch.utils.data.ConcatDataset([trainset, target_domain_dataset]), batch_size=config["batch"], shuffle=True, num_workers=16, drop_last=True)
testloader = DataLoader(testdataset, batch_size=config["batch"], shuffle=False, num_workers=16, drop_last=True)

# Define the DANN model and the optimizer
feature_extractor = LeNetFeatureExtractor(config=config)
label_predictor = LabelPredictor(num_classes=2,config=config)
domain_discriminator = DomainDiscriminator(config=config)
dann = DANN(feature_extractor, label_predictor, domain_discriminator)

optimizer = optim.Adam(dann.parameters(), lr=lr)

# Define the loss functions
clf_loss_fn = nn.CrossEntropyLoss()
domain_loss_fn = nn.BCELoss()

max_test_acc = 0
# Train the DANN model
for epoch in range(num_epochs):
    dann.train()
    num_correct_train = 0
    num_total_train = 0
    for i, (inputs, labels, domains) in enumerate(trainloader):
        # Set the domain labels (0 for source, 1 for target)
        # source_domain_labels = torch.zeros(inputs.size(0))
        # target_domain_labels = torch.ones(inputs.size(0))
        # domain_labels = torch.cat((source_domain_labels, target_domain_labels)).unsqueeze(1)
        domain_labels = domains.unsqueeze(1).float()
        # print("domain_labels: ",domain_labels.shape,type(domain_labels))

        # Zero the gradients
        optimizer.zero_grad()

        # Extract features and predict labels
        features = dann.feature_extractor(inputs)
        label_preds = dann.label_predictor(features)

        # Count Acc
        preds = F.softmax(label_preds, dim=1)
        pred_labels = torch.argmax(preds, dim=1)
        # print(pred_labels.shape,pred_labels,labels, labels.shape)
        num_correct_train += (pred_labels == labels).sum().item()
        num_total_train += labels.size(0)

        # Compute the label prediction loss
        clf_loss = clf_loss_fn(label_preds, labels)

        # Compute the domain classification loss
        domain_preds = dann.domain_discriminator(features)
        domain_loss = domain_loss_fn(domain_preds, domain_labels)

        # Compute the total loss and update the parameters
        total_loss = clf_loss + domain_loss
        total_loss.backward()
        optimizer.step()

        # Print the training statistics
        if (i+1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Clf Loss: {:.4f}, Domain Loss: {:.4f}, Total Loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(trainloader), clf_loss.item(), domain_loss.item(), total_loss.item()))
        
    print("Epoch [{}/{}], Train Accuracy: {:.2f}%".format(epoch+1, num_epochs, 100 * num_correct_train / num_total_train))

    # Evaluate the model on the test set
    dann.eval()
    test_acc = 0
    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for i, (inputs, labels, domains) in enumerate(testloader):
            features = dann.feature_extractor(inputs)
            logits = dann.label_predictor(features)
            preds = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(preds, dim=1)
            # print(pred_labels.shape,pred_labels,labels, labels.shape)
            num_correct += (pred_labels == labels).sum().item()
            num_total += labels.size(0)
        test_acc = 100 * num_correct / num_total
        max_test_acc = max(max_test_acc, test_acc)
        print("Epoch [{}/{}], Test Accuracy: {:.2f}%".format(epoch+1, num_epochs, max_test_acc))

        # Set the model back to training mode
        dann.train()
    
    writer.add_scalars('ACC', {"Train":num_correct_train / num_total_train,"Test":max_test_acc}, epoch)
