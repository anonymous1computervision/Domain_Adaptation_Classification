import matplotlib.pyplot as plt

plt.switch_backend('agg')


def plotAccuracy(x, y1, y2, figName, line1Name, line2Name, savePath):
    plt.figure(figsize=(20.48, 10.24))
    if y1:
        plt.plot(x, y1, linewidth=1.0, linestyle='-', label=line1Name)
    if y2:
        plt.plot(x, y2, linewidth=1.0, color='red', linestyle='--', label=line2Name)
    plt.title('Accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(savePath + figName + '_accuracy.png')
    plt.close()


def plotLoss(x, y1, y2, figName, line1Name, line2Name, savePath):
    plt.figure(figsize=(20.48, 10.24))
    if y1:
        plt.plot(x, y1, linewidth=1.0, linestyle='-', label=line1Name)
    if y2:
        plt.plot(x, y2, linewidth=1.0, color='red', linestyle='--', label=line2Name)
    plt.title('Loss')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(savePath + figName + '_loss.png')
    plt.close()
