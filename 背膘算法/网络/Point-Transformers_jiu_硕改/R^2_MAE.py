from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error
import numpy as np

labels = [13.8	,  12.2,	11.6,	11.1,	10.2,	10.8,	13.8,	11,	   15.8,	17.9	,16,	16,  	17.2,	18.4,	18.3]
outputs = [13.33,	11.78,	12.32,	12.53,	11.93,	12.14,	13.81,	12.12	,15.62,	20.12,	16.49,	16.47,	16.66,	16.1,	18.14]

print('真实值(labels)：', labels)
print('预测值：', outputs)

R2 = r2_score(y_true=labels, y_pred=outputs)
print('测试集R^2:{:.3f}'.format(R2))

MAE = mean_absolute_error(y_true=labels, y_pred=outputs)
print('测试集MAE:{:.3f}'.format(MAE))

MSE = mean_squared_error(y_true=labels, y_pred=outputs)
print('MSE:{:.3f}'.format(MSE))

# 计算RMSE
RMSE = np.sqrt(MSE)
print('RMSE:{:.3f}'.format(RMSE))

# 计算MAPE
MAPE = np.mean(np.abs((np.array(labels) - np.array(outputs)) / np.array(labels))) * 100
print('MAPE:{:.3f}'.format(MAPE))

# 计算SSE
SSE = np.sum((np.array(labels) - np.array(outputs)) ** 2)
print('SSE:{:.3f}'.format(SSE))

# 计算Explained Variance Score
EVS = explained_variance_score(y_true=labels, y_pred=outputs)
print('Explained Variance Score:{:.3f}'.format(EVS))

# 计算Median Absolute Error
MedAE = median_absolute_error(y_true=labels, y_pred=outputs)
print('Median Absolute Error:{:.3f}'.format(MedAE))