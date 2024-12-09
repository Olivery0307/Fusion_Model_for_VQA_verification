import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionFusionModel(nn.Module):
    def __init__(self, hidden_size=512):
        super(AttentionFusionModel, self).__init__()
        
        # Text model (BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers except the last two layers
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.text_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Image model (ResNet)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity() 
        
        # Freeze ResNet layers except the last block
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.image_model.layer4.parameters():
            param.requires_grad = True
        
        # Feature projection
        self.image_proj = nn.Linear(2048, hidden_size)
        self.text_proj = nn.Linear(768, hidden_size)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, images, questions, answers):
        images = images.to(device)
        image_features = self.image_model(images)  
        image_features = self.image_proj(image_features) 

        # Tokenization with proper device handling
        def process_text(text):
            tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=50).to(device)
            features = self.text_model(**tokens).pooler_output 
            return self.text_proj(features) 
        
        # Question and Answer Features
        question_features = process_text(questions) 
        answer_features = process_text(answers)
        
        # Combine Question and Answer Features
        combined_text_features = question_features + answer_features 
        combined_text_features = combined_text_features.unsqueeze(1) 
        image_features = image_features.unsqueeze(1)
        
        # Apply Attention Mechanism between text and image features
        combined_features, _ = self.attention(combined_text_features, image_features, image_features)
        combined_features = combined_features.squeeze(1) 

        # Concatenate features
        fused_features = torch.cat((combined_features, combined_text_features.squeeze(1)), dim=1)

        # Classification
        output = torch.sigmoid(self.classifier(fused_features))
        
        return output


class SimpleFusionModel(nn.Module):
    def __init__(self, hidden_size=512):
        super(SimpleFusionModel, self).__init__()
        
        # Text model (BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers except the last two layers
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.text_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        
        # Image model (ResNet)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()

        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in list(self.image_model.layer4.parameters()):
            param.requires_grad = True
        
        # Feature projection and normalization
        self.image_proj = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, images, questions, answers):
        # Ensure inputs are on the correct device
        images = images.to(device)
        
        #ResNet
        image_features = self.image_model(images) 
        image_features = self.image_proj(image_features)
        
        # Tokenization with proper device handling
        def process_text(text):
            tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=50).to(device)
            features = self.text_model(**tokens).pooler_output 
            return self.text_proj(features)
        
        # Question and Answer Features
        question_features = process_text(questions) 
        answer_features = process_text(answers) 
        
        # Combine Features
        combined_text_features = question_features + answer_features 
        combined_features = torch.cat((combined_text_features, image_features), dim=1) 
        
        # Classification
        output = torch.sigmoid(self.classifier(combined_features)) 
        
        return output

