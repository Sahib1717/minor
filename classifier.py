
# import numpy as np 
# import pandas as pd 
  
# #data visualisation libraries 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
# from pylab import rcParams 

# import torch 
# from torch.utils.data import DataLoader, TensorDataset 
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW 
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_score, precision_score, recall_score 
# from torch.cuda.amp import GradScaler, autocast 
# #to avoid warnings 
# import warnings 
# warnings.filterwarnings('ignore')
 
# data = pd.read_csv('../data/train.csv') 
# print(data.head()) 

# # Visualizing the class distribution of the 'label' column 
# column_labels = data.columns.tolist()[2:] 
# label_counts = data[column_labels].sum().sort_values() 
  
  
# # Create a black background for the plot 
# plt.figure(figsize=(7, 5)) 
  
# # Create a horizontal bar plot using Seaborn 
# ax = sns.barplot(x=label_counts.values, 
#                  y=label_counts.index, palette='viridis') 
  
  
# # Add labels and title to the plot 
# plt.xlabel('Number of Occurrences') 
# plt.ylabel('Labels') 
# plt.title('Distribution of Label Occurrences') 
  
# # Show the plot 
# # plt.show() 
# print(data[column_labels].sum().sort_values())

# # Create subsets based on toxic and clean comments 
# train_toxic = data[data[column_labels].sum(axis=1) > 0] 
# train_clean = data[data[column_labels].sum(axis=1) == 0] 
  
# # Number of toxic and clean comments 
# num_toxic = len(train_toxic) 
# num_clean = len(train_clean) 
  
# # Create a DataFrame for visualization 
# plot_data = pd.DataFrame( 
#     {'Category': ['Toxic', 'Clean'], 'Count': [num_toxic, num_clean]}) 
  
# # Create a black background for the plot 
# plt.figure(figsize=(7, 5)) 
  
# # Horizontal bar plot 
# ax = sns.barplot(x='Count', y='Category', data=plot_data, palette='viridis') 
  
  
# # Add labels and title to the plot 
# plt.xlabel('Number of Comments') 
# plt.ylabel('Category') 
# plt.title('Distribution of Toxic and Clean Comments') 
  
# # Set ticks' color to white 
# ax.tick_params() 
  
# # Show the plot 
# plt.show() 

# # Randomly sample 15,000 clean comments 
# train_clean_sampled = train_clean.sample(n=16225, random_state=42) 
  
# # Combine the toxic and sampled clean comments 
# dataframe = pd.concat([train_toxic, train_clean_sampled], axis=0) 
  
# # Shuffle the data to avoid any order bias during training 
# dataframe = dataframe.sample(frac=1, random_state=42) 
# print(train_toxic.shape) 
# print(train_clean_sampled.shape) 
# print(dataframe.shape) 
# train_texts, test_texts, train_labels, test_labels = train_test_split( 
#     dataframe['comment_text'], dataframe.iloc[:, 2:], test_size=0.25, random_state=42)
# test_texts, val_texts, test_labels, val_labels = train_test_split( 
#     test_texts, test_labels, test_size=0.5, random_state=42) 
# def tokenize_and_encode(tokenizer, comments, labels, max_length=128): 
#     # Initialize empty lists to store tokenized inputs and attention masks 
#     input_ids = [] 
#     attention_masks = [] 
  
#     # Iterate through each comment in the 'comments' list 
#     for comment in comments: 
  
#         # Tokenize and encode the comment using the BERT tokenizer 
#         encoded_dict = tokenizer.encode_plus( 
#             comment, 
  
#             # Add special tokens like [CLS] and [SEP] 
#             add_special_tokens=True, 
  
#             # Truncate or pad the comment to 'max_length' 
#             max_length=max_length, 
  
#             # Pad the comment to 'max_length' with zeros if needed 
#             pad_to_max_length=True, 
  
#             # Return attention mask to mask padded tokens 
#             return_attention_mask=True, 
  
#             # Return PyTorch tensors 
#             return_tensors='pt'
#         ) 
  
#         # Append the tokenized input and attention mask to their respective lists 
#         input_ids.append(encoded_dict['input_ids']) 
#         attention_masks.append(encoded_dict['attention_mask']) 
  
#     # Concatenate the tokenized inputs and attention masks into tensors 
#     input_ids = torch.cat(input_ids, dim=0) 
#     attention_masks = torch.cat(attention_masks, dim=0) 
  
#     # Convert the labels to a PyTorch tensor with the data type float32 
#     labels = torch.tensor(labels, dtype=torch.float32) 
  
#     # Return the tokenized inputs, attention masks, and labels as PyTorch tensors 
#     return input_ids, attention_masks, labels 
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
#                                           do_lower_case=True) 

# # Model Initialization 
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
#                                                       num_labels=6) 

# # Move model to GPU if available 
# device = torch.device( 
#     'cuda') if torch.cuda.is_available() else torch.device('cpu') 
# model = model.to(device) 

# # Tokenize and Encode the comments and labels for the training set 
# input_ids, attention_masks, labels = tokenize_and_encode( 
#     tokenizer, 
#     train_texts, 
#     train_labels.values 
# ) 
  
# # Tokenize and Encode the comments and labels for the test set 
# test_input_ids, test_attention_masks, test_labels = tokenize_and_encode( 
#     tokenizer, 
#     test_texts, 
#     test_labels.values 
# ) 
  
# # Tokenize and Encode the comments and labels for the validation set 
# val_input_ids, val_attention_masks, val_labels = tokenize_and_encode( 
#     tokenizer, 
#     val_texts, 
#     val_labels.values 
# ) 
  
  
# # print('Training Comments :',train_texts.shape) 
# # print('Input Ids         :',input_ids.shape) 
# # print('Attention Mask    :',attention_masks.shape) 
# # print('Labels            :',labels.shape)
# # k = 53
# # print('Training Comments -->>',train_texts.values[k]) 
# # print('\nInput Ids -->>\n',input_ids[k]) 
# # print('\nDecoded Ids -->>\n',tokenizer.decode(input_ids[k])) 
# # print('\nAttention Mask -->>\n',attention_masks[k]) 
# # print('\nLabels -->>',labels[k])

# # Creating DataLoader for the balanced dataset 
# batch_size = 32
# train_dataset = TensorDataset(input_ids, attention_masks, labels) 
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
  
# # testing set  
# test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels) 
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
  
# # validation set  
# val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels) 
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
# # print('Batch Size :',train_loader.batch_size) 
# Batch =next(iter(train_loader)) 
# # print('Each Input ids shape :',Batch[0].shape) 
# # print('Input ids :\n',Batch[0][0]) 
# # print('Corresponding Decoded text:\n',tokenizer.decode(Batch[0][0])) 
# # print('Corresponding Attention Mask :\n',Batch[1][0]) 
# # print('Corresponding Label:',Batch[2][0])
# optimizer = AdamW(model.parameters(), lr=2e-5)
# # Function to Train the Model 
# def train_model(model, train_loader, optimizer, device, num_epochs): 
#     # Loop through the specified number of epochs 
#     for epoch in range(num_epochs): 
#         # Set the model to training mode 
#         model.train() 
#         # Initialize total loss for the current epoch 
#         total_loss = 0
  
#         # Loop through the batches in the training data 
#         for batch in train_loader: 
#             input_ids, attention_mask, labels = [t.to(device) for t in batch] 
  
#             optimizer.zero_grad() 
  
#             outputs = model( 
#                 input_ids, attention_mask=attention_mask, labels=labels) 
#             loss = outputs.loss 
#             total_loss += loss.item() 
  
#             loss.backward() 
#             optimizer.step() 
  
#         model.eval()  # Set the model to evaluation mode 
#         val_loss = 0
  
#         # Disable gradient computation during validation 
#         with torch.no_grad(): 
#             for batch in val_loader: 
#                 input_ids, attention_mask, labels = [ 
#                     t.to(device) for t in batch] 
  
#                 outputs = model( 
#                     input_ids, attention_mask=attention_mask, labels=labels) 
#                 loss = outputs.loss 
#                 val_loss += loss.item() 
#         # Print the average loss for the current epoch 
#         print( 
#             f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)},Validation loss:{val_loss/len(val_loader)}') 
  
  
# # Call the function to train the model 
# train_model(model, train_loader, optimizer, device, num_epochs=1) 
# def evaluate_model(model, test_loader, device): 
#     model.eval()  # Set the model to evaluation mode 
  
#     true_labels = [] 
#     predicted_probs = [] 
  
#     with torch.no_grad(): 
#         for batch in test_loader: 
#             input_ids, attention_mask, labels = [t.to(device) for t in batch] 
  
#             # Get model's predictions 
#             outputs = model(input_ids, attention_mask=attention_mask) 
#             # Use sigmoid for multilabel classification 
#             predicted_probs_batch = torch.sigmoid(outputs.logits) 
#             predicted_probs.append(predicted_probs_batch.cpu().numpy()) 
  
#             true_labels_batch = labels.cpu().numpy() 
#             true_labels.append(true_labels_batch) 
  
#     # Combine predictions and labels for evaluation 
#     true_labels = np.concatenate(true_labels, axis=0) 
#     predicted_probs = np.concatenate(predicted_probs, axis=0) 
#     predicted_labels = (predicted_probs > 0.5).astype( 
#         int)  # Apply threshold for binary classification 
  
#     # Calculate evaluation metrics 
#     accuracy = accuracy_score(true_labels, predicted_labels) 
#     precision = precision_score(true_labels, predicted_labels, average='micro') 
#     recall = recall_score(true_labels, predicted_labels, average='micro') 
  
#     # Print the evaluation metrics 
#     print(f'Accuracy: {accuracy:.4f}') 
#     print(f'Precision: {precision:.4f}') 
#     print(f'Recall: {recall:.4f}') 
  
  
# # Call the function to evaluate the model on the test data 
# evaluate_model(model, test_loader, device) 

# # Save the tokenizer and model in the same directory 
# output_dir = "Saved_model"
# # Save model's state dictionary and configuration 
# model.save_pretrained(output_dir) 
# # Save tokenizer's configuration and vocabulary 
# tokenizer.save_pretrained(output_dir) 

# # Load the tokenizer and model from the saved directory 
# model_name = "Saved_model"
# Bert_Tokenizer = BertTokenizer.from_pretrained(model_name) 
# Bert_Model = BertForSequenceClassification.from_pretrained( 
#     model_name).to(device) 
# def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device): 
# 	user_input = [input_text] 

# 	user_encodings = tokenizer( 
# 		user_input, truncation=True, padding=True, return_tensors="pt") 

# 	user_dataset = TensorDataset( 
# 		user_encodings['input_ids'], user_encodings['attention_mask']) 

# 	user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False) 

# 	model.eval() 
# 	with torch.no_grad(): 
# 		for batch in user_loader: 
# 			input_ids, attention_mask = [t.to(device) for t in batch] 
# 			outputs = model(input_ids, attention_mask=attention_mask) 
# 			logits = outputs.logits 
# 			predictions = torch.sigmoid(logits) 

# 	predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int) 
# 	labels_list = ['toxic', 'severe_toxic', 'obscene', 
# 				'threat', 'insult', 'identity_hate'] 
# 	result = dict(zip(labels_list, predicted_labels[0])) 
# 	return result 


# text = 'Are you insane!'
# print(predict_user_input(input_text=text))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'  # You can change this to the appropriate model if needed
DistilBert_Model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=6)
DistilBert_Tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Ensure model is on the right device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DistilBert_Model.to(device)
# Load data
data = pd.read_csv('../data/train.csv')

# Use a smaller sample to reduce computation
data = data.sample(frac=0.2, random_state=42)

# Class labels
column_labels = data.columns.tolist()[2:]

# Balance dataset
train_toxic = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]
train_clean_sampled = train_clean.sample(n=len(train_toxic), random_state=42)
dataframe = pd.concat([train_toxic, train_clean_sampled], axis=0).sample(frac=1, random_state=42)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataframe['comment_text'], dataframe.iloc[:, 2:], test_size=0.25, random_state=42
)
test_texts, val_texts, test_labels, val_labels = train_test_split(
    test_texts, test_labels, test_size=0.5, random_state=42
)

# Use DistilBERT to reduce model size and computational power
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(column_labels))

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def tokenize_and_encode(tokenizer, texts, labels, max_length=64):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values, dtype=torch.float32)
    
    return input_ids, attention_masks, labels

# Tokenize and encode datasets
train_inputs, train_masks, train_labels = tokenize_and_encode(tokenizer, train_texts, train_labels)
val_inputs, val_masks, val_labels = tokenize_and_encode(tokenizer, val_texts, val_labels)
test_inputs, test_masks, test_labels = tokenize_and_encode(tokenizer, test_texts, test_labels)

# Data loaders
batch_size = 16  # Smaller batch size to reduce memory usage

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Mixed precision training using torch.cuda.amp
scaler = torch.cuda.amp.GradScaler()

def train_model(model, train_loader, val_loader, optimizer, device, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader):
            input_ids, attention_masks, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            # Forward pass with autocasting for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_masks, labels = [b.to(device) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                    total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_masks, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            predictions.append(logits.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).int().numpy()

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Train and evaluate the model
train_model(model, train_loader, val_loader, optimizer, device, epochs=3)
evaluate_model(model, test_loader, device)
def predict_user_input(input_text, model=DistilBert_Model, tokenizer=DistilBert_Tokenizer, device=device): 
    user_input = [input_text] 
  
    user_encodings = tokenizer( 
        user_input, truncation=True, padding=True, return_tensors="pt") 
  
    user_dataset = TensorDataset( 
        user_encodings['input_ids'], user_encodings['attention_mask']) 
  
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False) 
  
    model.eval() 
    with torch.no_grad(): 
        for batch in user_loader: 
            input_ids, attention_mask = [t.to(device) for t in batch] 
            outputs = model(input_ids, attention_mask=attention_mask) 
            logits = outputs.logits 
            predictions = torch.sigmoid(logits) 
  
    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int) 
    labels_list = ['toxic', 'severe_toxic', 'obscene', 
                   'threat', 'insult', 'identity_hate'] 
    result = dict(zip(labels_list, predicted_labels[0])) 
    return result 

# Test the prediction
text = 'Are you insane!'
print(predict_user_input(input_text=text))