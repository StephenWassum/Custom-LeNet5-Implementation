import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

directory = os.getcwd() 

output = os.path.join(directory, "P1outputImages")
os.makedirs(output, exist_ok=True)


class MNIST(Dataset):
    """
    Dataset class for loading MNIST data from parquet files.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        try:
            row = self.dataframe.iloc[index]
            imgBytes = row['image']['bytes']
            image = Image.open(io.BytesIO(imgBytes)).convert('L')

            imageNp = np.array(image, dtype=np.float32).reshape(28, 28)
            paddedImage = np.zeros((32, 32), dtype=np.float32)
            paddedImage[2:30, 2:30] = imageNp
            normalizedImage = (paddedImage / 127.5) - 1.0

            inputTensor = torch.from_numpy(normalizedImage).unsqueeze(0)
            label = int(row['label'])

            return inputTensor, label
        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise e


def generateDigitBitmaps():
    baseTemplates = {
        0: [
            "  ###      ",
            " #   #     ",
            " #   #     ",
            " #   #     ",
            " #   #     ",
            "  ###      ",
            "           "
        ],
        1: [
            "    #      ",
            "   ##      ",
            "    #      ",
            "    #      ",
            "    #      ",
            "    #      ",
            "           "
        ],
        2: [
            "  ###      ",
            " #   #     ",
            "     #     ",
            "   ##      ",
            "  #        ",
            " #####     ",
            "           "
        ],
        3: [
            "  ###      ",
            " #   #     ",
            "     #     ",
            "   ##      ",
            "     #     ",
            " #   #     ",
            "  ###      "
        ],
        4: [
            "    #      ",
            "   ##      ",
            "  # #      ",
            " #  #      ",
            " #####     ",
            "    #      ",
            "    #      "
        ],
        5: [
            " #####     ",
            " #         ",
            " ####      ",
            "     #     ",
            "     #     ",
            " #   #     ",
            "  ###      "
        ],
        6: [
            "  ###      ",
            " #         ",
            " #         ",
            " ####      ",
            " #   #     ",
            " #   #     ",
            "  ###      "
        ],
        7: [
            " #####     ",
            "     #     ",
            "    #      ",
            "    #      ",
            "   #       ",
            "   #       ",
            "           "
        ],
        8: [
            "  ###      ",
            " #   #     ",
            " #   #     ",
            "  ###      ",
            " #   #     ",
            " #   #     ",
            "  ###      "
        ],
        9: [
            "  ###      ",
            " #   #     ",
            " #   #     ",
            "  ####     ",
            "     #     ",
            "     #     ",
            "  ###      "
        ]
    }


    def pad_row(row):
        return row.ljust(12)

    bitmaps = []
    for digit in range(10):
        template = [pad_row(row) for row in baseTemplates[digit]]
        bitmap = [[1 if char == "#" else -1 for char in row] for row in template]
        bitmaps.append(torch.tensor(bitmap, dtype=torch.float32).flatten())

    return torch.stack(bitmaps)

class LeNet5Model(nn.Module):
    def __init__(self):
        super(LeNet5Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84) 

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))  
        return x


class StochasticDiagonalLevenbergMarquardt(optim.Optimizer):
    def __init__(self, parameters, lr=0.01, mu=1e-3, eta=1e-2):
        defaults = {'lr': lr, 'mu': mu, 'eta': eta}
        super(StochasticDiagonalLevenbergMarquardt, self).__init__(parameters, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if 'hessianDiag' not in state:
                    state['hessianDiag'] = torch.zeros_like(param.data)
                hessianDiag = state['hessianDiag']
                hessianDiag.add_(grad ** 2)
                adaptiveLr = lr / (mu + hessianDiag.sqrt())
                param.data -= adaptiveLr * grad

        return loss


class MAPLossFunction(nn.Module):
    def __init__(self, j: float = 0.1):
        super(MAPLossFunction, self).__init__()
        self.j = j

    def forward(self, outputs, targets):
        penalties = F.log_softmax(-outputs, dim=1)
        correctClassPenalty = penalties[range(len(targets)), targets]
        expOtherClasses = torch.exp(-penalties).sum(dim=1)
        jTensor = torch.tensor(self.j, device=outputs.device)  # Ensure compatibility with PyTorch tensors
        competitivePenalty = torch.log(torch.exp(-jTensor) + expOtherClasses)
        mapLoss = correctClassPenalty + competitivePenalty
        return mapLoss.mean()

def evaluateRbf(model, loader, targetVectors):
    model.eval()
    total = 0
    correct = 0
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            distances = torch.cdist(outputs, targetVectors)
            preds = distances.argmin(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predictions.append(preds.cpu())
            targets.append(labels.cpu())
    return correct / total, torch.cat(predictions), torch.cat(targets)


def train(model, trainLoader, testLoader, targetVectors, epochs, optimizer, lossFunction):
    trainErrors, testErrors = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        runningLoss = 0.0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            distances = torch.cdist(predictions, targetVectors)
            loss = lossFunction(-distances, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)

        trainAcc, _, _ = evaluateRbf(model, trainLoader, targetVectors)
        testAcc, testPredictions, testTargets = evaluateRbf(model, testLoader, targetVectors)

        trainErrors.append(1 - trainAcc)
        testErrors.append(1 - testAcc)

        print(f"Epoch {epoch}, Loss: {runningLoss / len(trainLoader.dataset):.4f}, Train Accuracy: {trainAcc:.4f}, Test Accuracy: {testAcc:.4f}")
        
    print("Training completed.")
    return trainErrors, testErrors, testPredictions, testTargets

def confusionMatrix(predictions, targets):
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    for prediction, target in zip(predictions, targets):
        matrix[target, prediction] += 1
    return matrix

def plotErrorRates(trainErrors, testErrors):
    plt.figure(figsize=(8, 6))
    plt.plot(trainErrors, label="Train Error")
    plt.plot(testErrors, label="Test Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title("Training and Test Error Rate")
    plt.legend()
    plt.savefig(os.path.join(output, "errorRatesChart.png"))
    plt.close()


def plotConfusionMatrix(testPredictions, testTargets):
    confusionMat = confusionMatrix(testPredictions, testTargets)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMat.numpy(), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output, "confusionMatrix.png"))
    plt.close()



def evaluateMisclassifiedExamples(model, testDataset, testTargets):
    misclassifiedCounts = {i: {j: 0 for j in range(10)} for i in range(10)}
    examples = {i: {j: None for j in range(10)} for i in range(10)}

    with torch.no_grad():
        for i in range(len(testDataset)):
            inputTensor, label = testDataset[i]
            inputTensor = inputTensor.unsqueeze(0).to(device)

            outputTensor = model(inputTensor)
            distances = torch.cdist(outputTensor, digitBitmaps)
            prediction = max(0, min(distances.argmin(dim=1).item(), 9))

            
            if prediction < 0 or prediction >= 10:
                print(f"Warning: Invalid prediction {prediction} for true label {label}. Skipping...")
                print(f"Sample {i}: True label: {label}, Prediction: {prediction}")
                print(f"Sample {i}: Distances shape: {distances.shape}")
                print(f"Sample {i}: Distances: {distances.squeeze().tolist()}")
                print(f"Sample {i}: Output tensor shape: {outputTensor.shape}")
                print(f"Sample {i}: Output tensor values (first 5): {outputTensor.squeeze()[:5].tolist()}")

                continue

            if prediction != label:
                misclassifiedCounts[label][prediction] += 1
                if examples[label][prediction] is None:
                    examples[label][prediction] = inputTensor.cpu()

    for trueLabel in range(10):
        maxMisclassified = max(misclassifiedCounts[trueLabel], key=misclassifiedCounts[trueLabel].get)
        count = misclassifiedCounts[trueLabel][maxMisclassified]

        if count == 0:
            print(f"No misclassifications for digit {trueLabel}")
        else:
            example = examples[trueLabel][maxMisclassified]
            plt.imshow(example.squeeze().numpy(), cmap="gray")
            plt.title(f"True: {trueLabel}, Pred: {maxMisclassified} ({count} times)")
            plt.savefig(os.path.join(output, f"mostMisclassified{trueLabel}_Pred{maxMisclassified}.png"))
            plt.close()
            
            
            
if __name__ == "__main__":
    trainingData, testingData = data.df_train, data.df_test
    trainDataset, testDataset = MNIST(trainingData), MNIST(testingData)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)

    digitBitmaps = generateDigitBitmaps().to(device)
       
    model = LeNet5Model().to(device)
    optimizer = StochasticDiagonalLevenbergMarquardt(model.parameters(), lr=0.01, mu=1e-3, eta=1e-2)
    lossFunction = MAPLossFunction(j=0.1)

    trainErrors, testErrors, testPredictions, testTargets = train(
        model, trainLoader, testLoader, digitBitmaps, epochs=20, optimizer=optimizer, lossFunction=lossFunction
    )
    
    plotErrorRates(trainErrors, testErrors)

    plotConfusionMatrix(testPredictions, testTargets)


    evaluateMisclassifiedExamples(model, testDataset, testTargets)
    
    torch.save(model.state_dict(), "LeNet5_1.pth")

    print(f"Final Train Error at epoch {len(trainErrors)}: {trainErrors[-1] * 100:.2f}%")
    print(f"Final Test Error at epoch {len(testErrors)}: {testErrors[-1] * 100:.2f}%")
