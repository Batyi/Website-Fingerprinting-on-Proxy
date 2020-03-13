import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# x = [3, 3, 3]
# y = [1, 3, 5]
# errors = [0.5, 0.25, 0.75]

# plt.figure()
# plt.errorbar(x, y, xerr=errors, fmt = 'o', color = 'k')
# plt.yticks((0, 1, 3, 5, 6), ('', 'x3', 'x2', 'x1',''))
# plt.show()

# [1,2,3,4,5,6,7,8,9,10]
# [1,2,3,4,5,6,7,8,9,10]
# [1,2,3,4,5,6,7,8,9,10]
# [1,2,3,4,5,6,7,8,9,10]


# numpy_data = np.array([[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8]])
# df_data = pd.DataFrame(data=numpy_data, columns=["row1", "row2", "row3", "row4", "row5", "row6", "row7", "row8"], index=["column1", "column2"])
# intialise data of lists. 
# data = {'Algorithm name': ['KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'krish', 'jack'], 'Age':[20, 21, 19, 18]} 
  
# Create DataFrame 
df = pd.read_csv("/home/batyi/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/machine_learning/results.csv") 
ax = sns.boxplot(x="Algorithm name", y="Accuracy", data=df)

# sns.boxplot(x="variable", y="value", data=pd.melt(df_data))
plt.title('Confidence Interval')
plt.show()