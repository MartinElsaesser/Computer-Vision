import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    # SLP
    import torch 
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.datasets as dsets
    import matplotlib.pylab as plt
    import numpy as np

    return dsets, nn, np, plt, torch, transforms


@app.cell
def _(dsets, transforms):
    ## Create and print the training dataset

    train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    print("Print the training dtatset:\n", train_dataset)
    return (train_dataset,)


@app.cell
def _(dsets, transforms):
    ## Create and print the validating dataset
    validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    print("Print the validation dtatset:\n", validation_dataset)
    return (validation_dataset,)


@app.cell
def _(plt, torch, train_dataset):
    ## Plot multiple training data
    h = plt.figure(figsize=(10,8))
    cols, rows = 5, 5
    for i in range(1, cols*rows+1):
        random_idx = torch.randint(len(train_dataset), size=(1,)).item() 
        img, label = train_dataset[random_idx]
        h.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()
    return


@app.cell
def _(nn):
    # Define SLP-Model
    class SLP(nn.Module):
        def __init__(self, input_size: int, output_size: int ) -> None:
            super(SLP, self).__init__()
            self.linear = nn.Linear(input_size, output_size, bias=False)
        # Prediction
        def forward(self, x):  # ty: ignore[invalid-mutable-override]
            y = self.linear(x)
            return y


    return (SLP,)


@app.cell
def _(torch):
    ## Set GPU if device is possible
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    if device !="cuda":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return (device,)


@app.cell
def _(SLP, device):
    ## Create Model
    input_dim: int = 28*28
    output_dim: int = 10
    model = SLP(input_dim, output_dim).to(device)
    print(model)
    return input_dim, model


@app.cell
def _(model):
    model.state_dict()
    return


@app.cell
def _(model, plt):
    ## Helper Function
    def PlotParameters(model): 
        W = model.state_dict()['linear.weight'].data.cpu()
        w_min = W.min().item()
        w_max = W.max().item()
        fig, axes = plt.subplots(2, 5)
        fig.subplots_adjust(hspace=0.01, wspace=0.1)
        for i, ax in enumerate(axes.flat):
            if i < 10:
        
                # Set the label for the sub-plot.
                ax.set_xlabel(f"class: {i}")

                # Plot the image.
                ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

                ax.set_xticks([])
                ax.set_yticks([])

            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
        plt.show()

    PlotParameters(model=model)
    return (PlotParameters,)


@app.cell
def _(model, nn, torch, train_dataset, validation_dataset):
    ## Define the learning_rate, optimizer, criterion and data loader
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100)
    return loss_func, optimizer, train_loader, validation_loader


@app.cell
def _(device, input_dim: int, nn, torch):
    loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    def train(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_func: nn.modules.loss._WeightedLoss, optimizer: torch.optim.Optimizer):
        size = len(dataloader.dataset)
        model.train()
        train_loss, correct = 0, 0
        for batchNr, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # set previously used gradient values to zero
            optimizer.zero_grad()

            # flatten input matrix and compute prediction error
            pred = model(X.view(-1, input_dim))
            loss = loss_func(pred, y)

            # Backpropagation
            loss.backward()  # Principle: param.grad += dloss/dparam
            optimizer.step() # Principle: param += -lr * param.grad

            # The following lines are calculating the accuracy values
            _, yhat = torch.max(pred.data, 1)
            correct += (yhat == y).sum().item()
            train_loss += loss
            if (batchNr+1) %100 == 0:
                loss, current = loss.item(), (batchNr+1)*len(y)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        accuracy = correct / size
        train_accuracy_list.append(accuracy)
        loss_list.append(train_loss.item()/size)


    return (
        loss_list,
        train,
        train_accuracy_list,
        val_accuracy_list,
        val_loss_list,
    )


@app.cell
def _(device, input_dim: int, nn, torch, val_accuracy_list, val_loss_list):
    # perform a prediction on the validationdata 
    def validate(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_func: nn.modules.loss._WeightedLoss):
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        #print(num_batches)
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for x_test, y_test in dataloader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                pred = model(x_test.view(-1, input_dim))
                val_loss += loss_func(pred, y_test)
                # The following lines are calculating the accuracy values
                _, yhat = torch.max(pred.data, 1)
                correct += (yhat == y_test).sum().item()
        val_loss /= num_batches  
        accuracy = correct / size
        val_accuracy_list.append(accuracy)
        val_loss_list.append(val_loss.item())
        print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f} Avg loss: {(val_loss.item()):>8f}")

    return (validate,)


@app.cell
def _(
    loss_func,
    model,
    optimizer,
    train,
    train_loader,
    validate,
    validation_loader,
):
    ## Training
    n_epochs = 30
    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------")
        train(train_loader, model, loss_func, optimizer)
        validate(validation_loader, model, loss_func)
    return


@app.cell
def _(
    PlotParameters,
    loss_list,
    model,
    plt,
    train_accuracy_list,
    val_accuracy_list,
    val_loss_list,
):
    ## Plot the loss and accuracy
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(loss_list,color=color, label="train loss")
    ax1.set_xlabel('epoch',color=color)
    ax1.set_ylabel('total loss',color=color)
    ax1.tick_params(axis='y', color=color)
    plt.legend()
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  
    ax2.plot( train_accuracy_list, color=color, label="train accuracy")
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()

    color = 'tab:orange'
    ax2.plot(val_accuracy_list, color=color, label="validation accuracy")

    color = 'tab:purple'
    ax2.plot(val_loss_list, color=color, label="validation loss")
    ax2.legend()

    PlotParameters(model)
    return


@app.cell
def _(device, input_dim: int, model, nn, np, plt, torch, validation_dataset):
    # Plot the data
    def show_data(data_sample):
        plt.imshow(data_sample[0].numpy().squeeze(), cmap='gray')
        plt.title('y = ' + str(data_sample[1]))
    
    Softmax_fn=nn.Softmax(dim=1)

    count, k = 0, -1
    for X, y in validation_dataset:
        k +=1
        X = X.to(device)
        y = torch.from_numpy(np.array([[y]]))
        y = y.to(device)
        z = model(X.reshape(-1, input_dim))
        _, yhat = torch.max(z, 1)
        yhat = int(yhat.item())
        if yhat != y:
            X, y = X.cpu(), y.cpu()
            show_data((X, y.item()))
            plt.show()
            print("yhat:", yhat)
            print("probability of class ", torch.max(Softmax_fn(z)).item())
            print('index: ', k)
            count += 1
        if count >= 5:
            break
        

    X, y = validation_dataset[63]
    show_data((X, y))
    Xtensor = X.to(device)
    y_hat = torch.max(Softmax_fn(model(Xtensor.reshape(-1, input_dim))))
    print(f'{y_hat=}')
    return Softmax_fn, X


@app.cell
def _(X, input_dim: int, model, torch, validation_dataset):
    _X, _y = validation_dataset[63]
    _X = _X.cpu()
    Weights = model.state_dict()['linear.weight'].data.cpu()
    # for bias
    # bias = model.state_dict()['linear.bias'].data.cpu()
    # torch.transpose(torch.matmul(Weights, X.view(input_dim, -1)), 1, 0) + bias
    # Softmax_fn(torch.transpose(torch.matmul(Weights, X.view(input_dim, -1)), 1, 0) + bias)

    # without bias
    pred = torch.transpose(torch.matmul(Weights, X.view(input_dim, -1)), 1, 0)

    return (pred,)


@app.cell
def _(Softmax_fn, pred):
    Softmax_fn(pred)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
