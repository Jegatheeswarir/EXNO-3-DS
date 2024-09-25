## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-09-25 100944](https://github.com/user-attachments/assets/253982e4-fd43-4ff4-ae5e-aa03e135c1a0)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-09-25 101208](https://github.com/user-attachments/assets/cd97c193-0294-4898-a186-9fa099edded1)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-09-25 101242](https://github.com/user-attachments/assets/912d565e-5a40-4b13-ba74-a52402ebb5d2)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-09-25 101330](https://github.com/user-attachments/assets/d2da3be9-2456-4817-a572-6861ba4c5bfb)
```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-09-25 101402](https://github.com/user-attachments/assets/78670082-7a6f-406d-bd9e-867bd3f48197)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-09-25 101535](https://github.com/user-attachments/assets/433aff06-72d0-4cfd-8698-4e49ad2a91d0)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![Screenshot 2024-09-25 101604](https://github.com/user-attachments/assets/5db7b62d-7c96-4067-848f-6150e70909fa)
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-09-25 101644](https://github.com/user-attachments/assets/f691536c-93a9-49c3-933a-4133a572fe2f)
```
df.skew()
```
![Screenshot 2024-09-25 101710](https://github.com/user-attachments/assets/65ef1284-b7cf-4de6-81e0-1bd161e1d762)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-09-25 101733](https://github.com/user-attachments/assets/822776bf-1195-472b-8e3c-cc8ee556fd2a)
```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df.skew()
```
![Screenshot 2024-09-25 101810](https://github.com/user-attachments/assets/51a1c78b-e175-4cc2-8b81-1b6f71889c8b)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-09-25 101845](https://github.com/user-attachments/assets/05b813cd-dc4f-4e84-b30a-19f9876905a0)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-25 101924](https://github.com/user-attachments/assets/3d977735-b0ae-4c5c-8fdb-3871c3b0dbce)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-09-25 101949](https://github.com/user-attachments/assets/3dea7587-bc94-40b4-8111-c60edf4291e8)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=892)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-25 102029](https://github.com/user-attachments/assets/9b9c51d4-a337-40c3-a560-af219098142f)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.       
