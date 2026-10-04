import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#
import ssl
import certifi
ssl._create_default_https_context= lambda: ssl.create_default_context(cafile=certifi.where())
#

class DigitCNN(nn.Module): # inherit all of pytorch's machinery (gradient calculation, knowing weights)
    def __init__(self):
        super().__init__()
        #First conv layer: looks for simple patterns (edges, strokes)
        #it slides 16 3x3 filters over the image, each filter is a tiny grid of learnable numbers that learn a curve
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels=16,kernel_size=3,padding=1)
        #Second conv layer: takes the feature maps and 
        self.conv2 =nn.Conv2d(in_channels= 16, out_channels=32,kernel_size=3,padding=1)
        #shrink the image by half each time, keep the strongest signals only
        self.pool = nn.MaxPool2d(kernel_size = 2,stride=2)
        #take the extracted features and decide the digit
        #after 2 pooling steps, the 28x28 image becomes a 7x7 image with 32 channels
        #so you have 1568 numbers feeding into 128 neurons
        self.fc1= nn.Linear(32*7*7,128) 
        #128 neurons producing 10 numbers(a score for every digit)
        self.fc2 = nn.Linear(128,10)
        #zero out 25% of the 128 neurons randomly each epoch
        self.dropout = nn.Dropout(0.25)
    def forward(self,x):
        x = self.conv1(x)     # Take the image and produce the 16 feature maps
        x = F.relu(x)         # Zero out negative values in the feature maps
        x = self.pool(x)      # Shrink the size of the image to 14x14

        x = self.conv2(x)     # Take the 14x14 image and produce the 32 feature maps from the 16 feature maps
        x = F.relu(x)         # Zero out negative values in the feature maps
        x = self.pool(x)      # Shrink the size of the image to 7x7

        x = x.view(x.size(0),-1) # Reshape the data into (how many images are in this batch, collapse everything else into single dimension)
        x = self.fc1(x)       # A linear layer computes a weighted sum of the 1568 input numbers and produces a single output , 128 times with 128 weights
        x = F.relu(x)         # Zero out the negative values in the 128 output
        x = self.dropout(x)   # Active only during training, zero out 25% of the 128 outputs randomly so no overfitting occurs.
        x = self.fc2(x)       # Computes a weighted sum of the 128 to produce a score for each of the digits(0-9), logits.
        return x              # Logits are compared to the true label on the image, loss is calculates, and after training, the highest becomes the network's prediction.








transform = transforms.Compose([ #takes a list of transformation steps and chains them into one object
    transforms.ToTensor(), #converts the raw image into a PyTorch tensor, and rescales the values into 0-1
    transforms.Normalize((0.1307,),(0.3081,))# subtract by the average mean and divide by the average spread  
])
#where to store the files, choose either the training images or the testing image, download the images from the internet.
#then attach the preprossing above to the data you chose.
train_data = datasets.MNIST(root="./data",train=True,download=True,transform =transform)
test_data = datasets.MNIST(root="./data",train=False,download=True,transform = transform)

print("Training examples: ",len(train_data))#check the sizes of the data
print("Test examples: ",len(test_data))

#fetch the first image, run it through the transform pipeline, and return a tuple [tensor, label]
#split the tuple into two variables
image,label = train_data[0] 

#report the tensors dimensions ([channels, pixels tall, pixels wide])
print("Image shape: ",image.shape)
#report the index of the image
print("Label: ",label)


#creates a 10 by 2 grid and returns the figure and individual plot slots
fig,axes = plt.subplots(1,5,figsize=(10,2))
for i in range(5):
    img, lbl = train_data[i]
    # draw the first image into the first slot, delete the channel dimension, and make the plot gray
    axes[i].imshow(img.squeeze(),cmap="gray")
    # write the number index of the image above the plot
    axes[i].set_title(f"Label: {lbl}")
    # hide the x/y axis ticks and numbers
    axes[i].axis("off")
# adjust layout
plt.tight_layout()
# save figure in file
plt.savefig("sample_mnist_images.png")
print("Saved sample_mnist_images.png - open it to check the images look right")   

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cpu") 
print(f"Using device: {device}")

# Data loader wraps the images into batches and shuffles them. 
train_loader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = 64,shuffle = False)

# Create the model with all it random weights moved to the used device
model = DigitCNN().to(device)

# Loss function that compares the Logits with the actual labels of the images and outputs a number (how wrong is the network)
criterion = nn.CrossEntropyLoss()

# Gradient descent function
# model.paramater() hands it the weights
# Adam adapts its step-size per-weight automatically
# lr is the learning rate
optimizer = optim.Adam(model.parameters(),lr= 0.001)

# An epoch is seeing all training images once
num_epochs = 5 

for epoch in range(num_epochs):
    model.train() # tells the model we're training, activate dropout
    running_loss = 0.0

    for images, labels in train_loader:
        images, label = images.to(device), labels.to(device)
        # Clear out old gradients before computing for this new batch
        optimizer.zero_grad() 

        # Run the batch through the network: conv1 > relu > pool > conv2 > relu > pool > flatten > fc1 > relu > dropout> fc2
        # This produces 64 sets of 10 logits
        outputs = model(images)

        # Compare the 64 sets of 10 logits with the 64 labels
        # This produces 1 loss number for the batch
        loss = criterion(outputs, labels)

        # Compute the gradient (each weight's contribution to the loss)
        loss.backward()
        # Update the weights based on the gradients
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluate on the test set
model.eval() # Deactivate the dropout
correct = 0
total = 0

with torch.no_grad(): # No need to compute gradients when just evaluating(checking accuracy)
    for images, labels in test_loader:
        images, labels = images.to(device),labels.to(device)

        outputs = model(images)

        # For each image, finds which of the 10 logits is highest, and returns its index(the predicted digit)
        _, predicted = torch.max(outputs,1)

        # Total images is added each iteration
        total += labels.size(0)
        # Number of correct prediction summed and added 
        correct += (predicted == labels).sum().item()
accuracy = 100*(correct /total)
print(f"Test accuracy: {accuracy: .2f}%")

# Saves all the trained weights to a file
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved to mnist_cnn.pth")