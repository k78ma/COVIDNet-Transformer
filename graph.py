from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":

    N = 4
    width = 0.5
    margin = (1 - width) + width / 2

    data = {
        'Supervised': 0.993, 
        'Self-Supervised': 0.994, 

        'Supervised (AUC)' :  0.996,
        'Self-Supvervised (AUC)': 0.994,

    }
    std = [0.002057, 0.0010446, 0.001500, 0.001164]

    keys = list(['Supervised', 'Self-Supervised', ' Supervised ', ' Self-Supervised '])
    values = list([0.993, 0.994, 0.996, 0.994])
    
    _, ax = plt.subplots(figsize=(12,8))
    # creating the bar plot
    bar_plot = plt.bar(keys, values, color=['royalblue', 'royalblue', 'darkorange', 'darkorange'], yerr=std, width=width, capsize=10, ecolor='black', edgecolor='black')
    
    blue_patch = mpatches.Patch(color='royalblue', label='CE Optimization')
    orange_patch = mpatches.Patch(color='darkorange', label='AUC Maximization')
    plt.legend(handles=[blue_patch, orange_patch], prop={'size': 15})

    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. - 0.05, values[idx],
                    values[idx],
                    ha='right', va='bottom', rotation=0, fontsize=15)
    
    plt.xlim(-margin, N - 1 + margin)
    plt.ylim([0.980, 1.0])
    plt.xlabel("Pre-trained Model", fontsize=18, labelpad=20)
    plt.xticks(fontsize=15)
    plt.ylabel("Area Under the Curve (AUC)", fontsize=18, labelpad=20)
    plt.yticks(fontsize=15)
    autolabel(bar_plot)
    plt.savefig('auc-score.png')