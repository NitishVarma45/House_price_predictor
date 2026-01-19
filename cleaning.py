import numpy as np
import pandas as pd


#Reading the file.
data = pd.read_csv("Bengaluru_House_Data.csv")
#Dimesions of the dataset.
print(data.head())
print(data.shape)
#Value_Counts.
for columns in data:
    print(data[columns].value_counts())
    print("_"*25)

#Checking for null values.
print(data.isna().sum())
#Dropping columns with more null values.
#removing area_type as its just types of measuring a flat.
data.drop(columns=['area_type','availability','society','balcony'], inplace=True)

print(data.info())
'''
print(data['location'].value_counts())
print(data['size'].value_counts())
'''
#As there is only one missing value in location, we can replace it with the highest occured location.
data['location'] = data['location'].fillna('Whitefield')
#Same for size.
data['size'] = data['size'].fillna('2 BHK')
#No nulls in total_sqft.
#Nulls in bath are being replaced with the median of 'bath'.
data['bath'] = data['bath'].fillna(data['bath'].median())
#No nulls in proce.
#new bhk column
#converted 1 bhk / 1 bedroom to just 1.
data['bhk'] = data['size'].str.split().str.get(0).astype(int)

#there are ranges in sqft like 1221 - 1334, for which we will write a function to take their average.
def conversion_range(x):
    temp = x.split('-')
    if(len(temp)==2):
        return (float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
data['total_sqft'] = data['total_sqft'].apply(conversion_range) 

data.drop(columns='size',inplace=True)

#adding price per sqft.
data['price_per_sqft'] = data['price']*100000 / data['total_sqft']

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()
location_count_10 = location_count[location_count<=10]
data['location'] = data['location'].apply(lambda x : 'other' if x in location_count_10 else x)
print(data.head())

#removing outliers.
data = data[((data['total_sqft'] / data['bhk']) >= 300)]

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

data = remove_outliers_sqft(data)

def bhk_outlier_remover(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values
                )

    return df.drop(exclude_indices, axis='index')
data = bhk_outlier_remover(data)
data.drop(columns = 'price_per_sqft',inplace = True)
data.to_csv('cleandata.csv', index=False)
