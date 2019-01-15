import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###Data import
data = pd.read_csv('D:/Desktop/data.csv', delimiter=",",dtype=None)
data.iat[534953,5]=3525.859345 #Median replace missing value

#Dataframe with only streams using p2p technology
data_with_p2p=data[data.connected==True]

#Group by for each feature
data_isp=data.groupby('isp').mean()
data_stream=data.groupby('#stream').mean()
data_browser=data.groupby('browser').mean()
data_connected=data.groupby('connected').mean()

#Group by for each feature in data_with_p2p
data_isp_p2p=data_with_p2p.groupby('isp').mean()
data_stream_p2p=data_with_p2p.groupby('#stream').mean()
data_browser_p2p=data_with_p2p.groupby('browser').mean()

#Calculate ratio between p2p and cdn (p2p / cdn)
datas=[('data_isp',data_isp),('data_stream',data_stream),('data_browser',data_browser),('data_connected',data_connected),('data_isp_p2p',data_isp_p2p),('data_stream_p2p',data_stream_p2p),('data_browser_p2p',data_browser_p2p)]

mean_var=[]
for k in range(len(datas)):
    datas[k][1]['total down']=datas[k][1].p2p+datas[k][1].cdn   
    datas[k][1]['ratio']=datas[k][1].p2p/(datas[k][1].p2p+datas[k][1].cdn)
    print(datas[k][0],'ratio var:',datas[k][1].ratio.var()) #ratio variance mean 
    print(datas[k][0],'ratio mean:',datas[k][1].ratio.mean())
    mean_var.append([datas[k][1].ratio.mean(),datas[k][1].ratio.var()])

Mean_var=pd.DataFrame(mean_var, columns = ['Mean', 'Var'],index =[datas[k][0] for k in range(len(datas))])

Mean_var['Std']=Mean_var.Var.apply(lambda x: np.sqrt(x))
Mean_var['Inf']=Mean_var.Mean-Mean_var.Std
Mean_var['Max']=Mean_var.Mean+Mean_var.Std

data1=data[data.cdn<0.4*10**8]
data2=data[(data.cdn>=0.4*10**8) & (data.cdn<0.9*10**8)]
data3=data[data.cdn>=0.9*10**8]

data123=[data1,data2,data3]
Data123=[]

for k in range(3):
    data123[k]['total down']=data123[k].p2p+data123[k].cdn  
    data123[k]['ratio']=data123[k].p2p/(data123[k].p2p+data123[k].cdn)
    
###Data visualisation

plt.clf()

plt.figure(1)
plt.title('Ratio for p2p connected streams')
plt.ylabel('ratio')
plt.bar(x=data_isp_p2p.index.tolist(),height=data_isp_p2p.ratio)
plt.xlabel('ISP')
plt.savefig('D:/Documents/streamroot/img/ratio_isp_connected.png')

plt.figure(10)
plt.bar(x=data_stream_p2p.index.tolist(),height=data_stream_p2p.ratio)
plt.title('Ratio for p2p connected streams')
plt.xlabel('Stream')
plt.ylabel('ratio')
plt.savefig('D:/Documents/streamroot/img/ratio_stream_connected.png')


plt.figure(11)
plt.bar(x=data_browser_p2p.index.tolist(),height=data_browser_p2p.ratio)
plt.title('Ratio for p2p connected streams')
plt.xlabel('Browser')
plt.ylabel('ratio')
plt.savefig('D:/Documents/streamroot/img/ratio_browser_connected.png')

log_data_cdn=np.log1p(data.cdn)    
log_data_p2p=np.log1p(data.p2p)

plt.figure(2)
sns.distplot(log_data_p2p,color='blue',norm_hist="True")
sns.distplot(log_data_cdn,color='green',norm_hist="True")
plt.gca().legend(('p2p','cdn'))
plt.title('Distribution of log(p2p) and log(cdn)')
plt.xlabel('log(Download) Ko')
plt.savefig('D:/Documents/streamroot/img/log_data_p2p_cdn.png')

plt.figure(17)
sns.distplot(data.p2p,color='blue',norm_hist="True")
sns.distplot(data.cdn,color='green',norm_hist="True")
plt.gca().legend(('p2p','cdn'))
plt.title('Distribution of p2p and cdn')
plt.xlabel('Download Ko')
plt.savefig('D:/Documents/streamroot/img/distrib_data_p2p_cdn.png')

plt.figure(3)
sns.distplot(log_data_p2p,color='blue',norm_hist="True")
plt.title('Distribution of log(p2p)')
plt.savefig('D:/Documents/streamroot/img/log_data_p2p.png')

plt.figure(4)
sns.distplot(log_data_cdn,color='green',norm_hist="True")
plt.title('Distribution of log(cdn)')
plt.savefig('D:/Documents/streamroot/img/log_data_cdn.png')

plt.figure(5)
plt.title('Pearson correlation matrix')
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.savefig('D:/Documents/streamroot/img/correlation_matrix.png')

plt.figure(6)
plt.bar(x=data_isp_p2p.index.tolist(),height=data_isp_p2p['total down'])
plt.bar(x=data_isp_p2p.index.tolist(),height=data_isp_p2p.p2p)
plt.ylabel('volume download (Ko)')
plt.title('Total volume downloaded by ISP')
plt.gca().legend(('cdn','p2p'))
plt.savefig('D:/Documents/streamroot/img/download_isp_connected.png')

plt.figure(7)
plt.bar(x=data_browser_p2p.index.tolist(),height=data_browser_p2p['total down'])
plt.bar(x=data_browser_p2p.index.tolist(),height=data_browser_p2p.p2p)
plt.title('Total volume downloaded by Browser')
plt.ylabel('volume download (Ko)')
plt.gca().legend(('cdn','p2p'))
plt.savefig('D:/Documents/streamroot/img/download_browser_connected.png')

plt.figure(8)
plt.bar(x=data_stream_p2p.index.tolist(),height=data_stream_p2p['total down'])
plt.bar(x=data_stream_p2p.index.tolist(),height=data_stream_p2p.p2p)
plt.ylabel('volume download (Ko)')
plt.gca().legend(('cdn','p2p'))
plt.title('Total volume downloaded by Browser connected to p2p')
plt.savefig('D:/Documents/streamroot/img/download_stream_connected.png')

plt.figure(20)
plt.title('Total volume downloaded')
plt.ylabel('Volume downloaded (Ko)')
plt.bar(x=['data1','data2','data3'],height=[(data1.cdn+data1.p2p).sum(),(data2.cdn+data2.p2p).sum(),(data3.cdn+data3.p2p).sum()])
plt.savefig('D:/Documents/streamroot/img/volume_down.png')

plt.figure(21)
plt.title('Number of streams')
plt.bar(x=['data1','data2','data3'],height=[data1.cdn.count(),data2.cdn.count(),data3.cdn.count()])
plt.savefig('D:/Documents/streamroot/img/count_stream.png')

plt.figure(22)
plt.title('Ratio by volume')
plt.bar(x=['data1','data2','data3'],height=[data1.p2p.sum() / (data1.cdn+data1.p2p).sum(),data2.p2p.sum() / (data2.cdn+data2.p2p).sum(),data3.p2p.sum() / (data3.cdn+data3.p2p).sum()])
plt.ylabel('Ratio')
plt.savefig('D:/Documents/streamroot/img/ratio_volume.png')

plt.figure(23)
plt.bar(x=data.groupby('browser').count().index.tolist(),height=data.groupby('browser').connected.count())
plt.title('Number of stream')
plt.xlabel('Browser')
plt.savefig('D:/Documents/streamroot/img/count_browser.png')

plt.figure(24)
plt.bar(x=data.groupby('isp').count().index.tolist(),height=data.groupby('isp').connected.count())
plt.title('Number of stream')
plt.xlabel('ISP')
plt.savefig('D:/Documents/streamroot/img/count_isp.png')

plt.figure(25)
plt.bar(x=data1.groupby('isp').count().index.tolist(),height=data1.groupby('isp').mean().ratio)
plt.title('Ratio for data1 (low volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_low.png')

plt.figure(26)
plt.bar(x=data2.groupby('isp').count().index.tolist(),height=data2.groupby('isp').mean().ratio)
plt.title('Ratio for data2 (medium volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_medium.png')

plt.figure(27)
plt.bar(x=data3.groupby('isp').count().index.tolist(),height=data3.groupby('isp').mean().ratio)
plt.title('Ratio for data3 (high volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_high.png')

plt.figure(28)
plt.bar(x=data1.groupby('browser').count().index.tolist(),height=data1.groupby('browser').mean().ratio)
plt.title('Ratio for data1 (low volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_br_low.png')

plt.figure(29)
plt.bar(x=data2.groupby('browser').count().index.tolist(),height=data2.groupby('browser').mean().ratio)
plt.title('Ratio for data2 (medium volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_br_medium.png')

plt.figure(30)
plt.bar(x=data3.groupby('browser').count().index.tolist(),height=data3.groupby('browser').mean().ratio)
plt.title('Ratio for data3 (high volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_br_high.png')

plt.figure(31)
plt.bar(x=data1.groupby('#stream').count().index.tolist(),height=data1.groupby('#stream').mean().ratio)
plt.title('Ratio for data1 (low volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_st_low.png')

plt.figure(32)
plt.bar(x=data2.groupby('#stream').count().index.tolist(),height=data2.groupby('#stream').mean().ratio)
plt.title('Ratio for data2 (medium volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_st_medium.png')

plt.figure(33)
plt.bar(x=data3.groupby('#stream').count().index.tolist(),height=data3.groupby('#stream').mean().ratio)
plt.title('Ratio for data3 (high volume)')
plt.savefig('D:/Documents/streamroot/img/ratio_st_high.png')









    




