# Labels:
# 0=Normal
# 1=Bacteria
# 2=Virus
# 3=Covid-19

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.245, ],
    std=[0.234, ]
)

transform = transforms.Compose([transforms.Resize(500),
                                transforms.CenterCrop(500),
                                transforms.ToTensor(),
                                normalize])

classes = ('NORMAL', 'BACTERIA', 'VIRUS', 'COVID-19')

trainSet = torchvision.datasets.ImageFolder(root="Train", transform=transform)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=2)

testSet = torchvision.datasets.ImageFolder(root="Test", transform=transform)

testLoader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=2)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.con1 = nn.Conv2d(3, 6, 4)
        self.con2 = nn.Conv2d(6, 10, 4)
        self.con3 = nn.Conv2d(10, 10, 4)
        self.con4 = nn.Conv2d(10, 13, 4)
        self.con5 = nn.Conv2d(13, 15, 4)
        self.con6 = nn.Conv2d(15, 64, 4)

        # kill neuron with a probability of 20%
        self.conDrop = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(200704, 4)

        self.MaxPool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = self.MaxPool(x)
        x = self.conDrop(x)
        x = F.relu6(self.con2(x))
        x = self.conDrop(x)
        x = F.relu6(self.con3(x))
        x = self.MaxPool(F.relu(self.con4(x)))
        x = F.relu(self.con5(x))
        x = self.conDrop(x)
        x = self.MaxPool(F.relu6(self.con6(x)))

        x = x.view(-1, 200704)
        x = self.fc1(x)
        return x

def Train(lr, epochs, epochSize):

    # Set model to training mode
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):

            modelInput, label = data

            optimizer.zero_grad()

            output = model(modelInput)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            lossList.append(loss.item())

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            if i == epochSize:
                break


def Test(testMode):
    # Set Model in Test mode
    model.eval()
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    AIPrediction = [0, 0, 0, 0]
    with torch.no_grad():
        x = 0
        for data in testLoader:
            x +=1
            modelInput, label = data
            output = model(modelInput)

            _, predicted = torch.max(output, 1)
            if predicted == label:
                class_correct[label] += 1
            class_total[label] += 1
            AIPrediction[predicted.data[0]] += 1
            if x == 500:
                break


    accuracy = (100 * (class_correct[0]+class_correct[1]+class_correct[2]+class_correct[3]) / (class_total[0]+class_total[1]+class_total[2]+class_total[3]))

    if testMode == "d":
        for i in range(4):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            print(class_correct[i], "-", class_total[i])
            print("AI prediction count:", AIPrediction[i])
    print("The accuracy is %3d %%" % accuracy)

    return accuracy


def saveModel(name, epoch, imageCount):
    torch.save(model, "saved_models/" + str(name) + "-EPOCH:" + str(epoch) + "-IMAGES:" + str(imageCount) + ".pt")

def loadModel(name, epoch, imageCount):
    if os.path.isfile("saved_models/" + name + "-EPOCH:" + str(epoch) + "-IMAGES:" + str(imageCount) + ".pt"):
        return torch.load("saved_models/" + name + "-EPOCH:" + str(epoch) + "-IMAGES:" + str(imageCount) + ".pt")

def saveVariableToFile(variable, fileName, directory):
    with open(directory+"/"+fileName+".txt","w") as file:
        file.write(str(variable))
        
#test mode d is for a detaild loss analysis, for not detail anything else 
def main(epochs, epochSize, lr, modelName, testMode):
    accuracy = []
    for x in range(epochs):
        Train(lr, 1, epochSize)
        saveModel(modelName, x+1, epochSize)
        accuracy.append(Test(testMode))
    saveVariableToFile(accuracy, "accuracy-of-" + modelName, "accuracyLists")
    saveVariableToFile(lossList, "loss-of-" + modelName, "LossLists")

model = Net()
lossList = []
model = loadModel("LungNet V.1.0",5,1000)
main(100, 1000, 0.00001, "lungNet.V.1.1", "d")
