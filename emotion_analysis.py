import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# get a list of files
f_list = glob.glob('./Output/vgg19*.csv')

# files are not sorted. use dict to add key value by csv info_xx name
f_list_dict = {}
for f_name in f_list:
    f_list_dict[int(f_name.split('_')[-1].split('.')[0].split('o')[-1])] = f_name

# will hold emotion percent totals of each videp
emotion_totals_list = []

# plot rolling mean of each video emotion
for i in range(10):
    f_name = f_list_dict.get(i + 1)

    df = pd.read_csv(f_name)
    df = df.loc[:,df.columns!='Unnamed: 0'] # remov unnamed emotions
    df_emotions = df.loc[df.sum(axis=1) != 1,:].reset_index(drop=True) # remove rows with single emotions
    df_emotions_rm = df_emotions.rolling(30).mean()
    
    plt.stackplot(df_emotions_rm.index, df_emotions_rm.Angry, df_emotions_rm.Disgust, df_emotions_rm.Fear,
                  df_emotions_rm.Happy, df_emotions_rm.Neutral, df_emotions_rm.Sad, df_emotions_rm.Surprise,
                  labels=df_emotions_rm.columns)
    plt.legend(loc='upper left')
    plt.xlabel('Sec')
    plt.ylabel('Rolling Mean')
    plt.title('Rolling Mean Every 30 Seconds of Group Emotions Vid {}'.format(i+1))
    plt.savefig('./Output/GroupEmotions_{}.png'.format(i+1))
    plt.show()
    plt.close()

    # append totals to emotion totals list
    emotion_totals_list.append((df_emotions.sum(axis=0)/df_emotions.sum(axis=0).sum()).to_list())

# create heat map of percent totals of each video
df_emotion_totals = pd.DataFrame(emotion_totals_list, columns=df.columns, index=[i+1 for i in range(10)])
sns.heatmap(df_emotion_totals, annot = True)
plt.title('Percent Total of Emotions for Video Dataset')
plt.savefig('./Output/heatmap.png')
plt.show()

# create stacked percentage chart to compare models
model_comparison_dict = {'Resnet' : './Output/resnet_frame_info2.csv',
                         'VGG11' : './Output/vgg11_frame_info2.csv',
                         'VGG19' : './Output/vgg19_frame_info2.csv'}

model_emotion_totals_list = []
for model in model_comparison_dict.keys():
    f_name = model_comparison_dict.get(model)
    df = pd.read_csv(f_name)
    df = df.loc[:, df.columns != 'Unnamed: 0']
    model_emotion_totals_list.append(df.sum(axis=0))

df_models = pd.DataFrame(model_emotion_totals_list, index=model_comparison_dict.keys(),columns=df.columns)

resnet = df_models.loc['Resnet',:].to_list()
vgg11 = df_models.loc['VGG11',:].to_list()
vgg19 = df_models.loc['VGG19',:].to_list()
labels = df.columns

width = 0.25
x = np.arange(len(df_models.columns))
fig, ax = plt.subplots()
bar1 = ax.bar(x - (4 * width)/ 6, resnet, width, label='resnet')
bar2 = ax.bar(x - (4 * width)/ 6, vgg11, width, label='vg11')
bar3 = ax.bar(x + width / 3, vgg19, width, label='vg19')

ax.set_ylabel('Total')
ax.set_title('Emotion Totals by Model For Video 2')
ax.set_xticks(x, labels)
ax.legend()


fig.tight_layout()
plt.savefig('./Output/model_comparions.png')
plt.show()


