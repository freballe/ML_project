from matplotlib import pyplot as plt

def plot_results_acc(results):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    for i in range(1, len(results) + 1):
        ax = fig.add_subplot(6, 5, i)
        history = results[str(i-1)]["histories"][0]      # 0 for history of first round 
        x = range(0, len(history["val_acc"]))
        y = history["val_acc"]
        y2 = history["acc"]
        ax.plot(x, y, 'r-', markersize=3, label="Validation accuracy")
        ax.plot(x, y2, 'b-', markersize=3, label="Train accuracy")
        ax.title.set_text(str(i))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.savefig('accuracy.png')
    plt.show()


def plot_results_loss(results):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    for i in range(1, len(results) + 1):
        ax = fig.add_subplot(6, 5, i)
        history = results[str(i-1)]["histories"][0]      # 0 for history of first round 
        x = range(0, len(history["val_loss"]))
        y = history["val_loss"]
        y2 = history["loss"]
        ax.plot(x, y, 'r-', markersize=3, label="Validation loss")
        ax.plot(x, y2, 'b-', markersize=3, label="Train loss")
        ax.title.set_text(str(i))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.savefig('loss.png')
    plt.show()