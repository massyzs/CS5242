import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LeNet architecture as the feature extractor
# class LeNetFeatureExtractor(nn.Module):
#     def __init__(self, config):
#         super(LeNetFeatureExtractor, self).__init__()
#         self.config = config

#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)

#         if self.config["norm_type"] == "BN":
#             self.norm1 = nn.BatchNorm2d(6)
#             self.norm2 = nn.BatchNorm2d(16)
#         elif self.config["norm_type"] == "LN":
#             self.norm1 = nn.LayerNorm([6, 124, 124])
#             self.norm2 = nn.LayerNorm([16, 58, 58])

#         if self.config["activation"] == "relu":
#             self.activation = nn.ReLU(inplace=True)
#         elif self.config["activation"] == "leaky_relu":
#             self.activation = nn.LeakyReLU(inplace=True)
#         elif self.config["activation"] == "sigmoid":
#             self.activation = nn.Sigmoid()
#         elif self.config["activation"] == "tanh":
#             self.activation = nn.Tanh()

#         self.dropout1 = nn.Dropout(p=config["dropout_rate"])
#         self.dropout2 = nn.Dropout(p=config["dropout_rate"])

#     def forward(self, x):
#         out = self.conv1(x)

#         if self.config["norm"]:
#             out = self.norm1(out)

#         out = self.activation(out)
#         out = F.max_pool2d(out, 2)

#         if self.config["dropout"]:
#             out = self.dropout1(out)  # Applying dropout after max pooling

#         out = self.conv2(out)

#         if self.config["norm"]:
#             out = self.norm2(out)

#         out = self.activation(out)
#         out = F.max_pool2d(out, 2)

#         if self.config["dropout"]:
#             out = self.dropout2(out)  # Applying dropout after max pooling

#         out = out.view(self.config["batch"], -1)
#         return out

class LeNetFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(LeNetFeatureExtractor, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        if self.config["norm_type"] == "BN":
            self.norm1 = nn.BatchNorm2d(6)
            self.norm2 = nn.BatchNorm2d(16)
        elif self.config["norm_type"] == "LN":
            self.norm1 = nn.LayerNorm([6, 124, 124])
            self.norm2 = nn.LayerNorm([16, 58, 58])

        if self.config["activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.config["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif self.config["activation"] == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.config["activation"] == "tanh":
            self.activation = nn.Tanh()

        self.dropout1 = nn.Dropout(p=config["dropout_rate"])
        self.dropout2 = nn.Dropout(p=config["dropout_rate"])

    def forward(self, x):
        out = self.conv1(x)

        if self.config["norm"]:
            out = self.norm1(out)

        out = self.activation(out)
        out = F.max_pool2d(out, 2)

        if self.config["dropout"]:
            out = self.dropout1(out)  # Applying dropout after max pooling

        out = self.conv2(out)

        if self.config["norm"]:
            out = self.norm2(out)

        out = self.activation(out)
        out = F.max_pool2d(out, 2)

        if self.config["dropout"]:
            out = self.dropout2(out)  # Applying dropout after max pooling

        out = out.view(self.config["batch"], -1)
        # print("out: ",out.shape)
        return out


# Define the label predictor
class LabelPredictor(nn.Module):
    def __init__(self, num_classes, config):
        super(LabelPredictor, self).__init__()
        self.config = config

        self.fc1   = nn.Linear(13456, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

        if self.config["norm_type"]=="BN":
            self.norm3=nn.BatchNorm1d(120)
            self.norm4=nn.BatchNorm1d(84)
        elif self.config["norm_type"]=="LN":
            self.norm3=nn.LayerNorm(120)
            self.norm4=nn.LayerNorm(84)
        
        if self.config["activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.config["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif self.config["activation"] == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.config["activation"] == "tanh":
            self.activation = nn.Tanh()

        self.dropout3 = nn.Dropout(p=config["dropout_rate"])
        self.dropout4 = nn.Dropout(p=config["dropout_rate"])

    def forward(self, x):
        out=self.fc1(x)
        out = self.activation(out)
        if self.config["norm"]:
            out=self.norm3(out)
        if self.config["dropout"]:
            out = self.dropout3(out) # Applying dropout after ReLU

        out=self.fc2(out)
        if self.config["norm"]:
            out=self.norm4(out)
        out = self.activation(out)
        if self.config["dropout"]:
            out = self.dropout4(out) # Applying dropout after ReLU

        out = self.fc3(out)
        return out

# Define the domain discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, config):
        super(DomainDiscriminator, self).__init__()
        self.config = config
        self.fc1   = nn.Linear(13456, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class DANN(nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_discriminator):
        super(DANN, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_discriminator = domain_discriminator

    def forward(self, x, mode="predict"):
        features = self.feature_extractor(x)
        if mode == "predict":
            return self.label_predictor(features)
        elif mode == "discriminate":
            return self.domain_discriminator(features)
        else:
            raise ValueError("Invalid mode. Must be 'predict' or 'discriminate'.")


# # Example usage
# feature_extractor = LeNetFeatureExtractor()
# label_predictor = LabelPredictor(num_classes=2)
# domain_discriminator = DomainDiscriminator()
# dann = DANN(feature_extractor, label_predictor, domain_discriminator)
