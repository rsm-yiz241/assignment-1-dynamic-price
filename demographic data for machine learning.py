#read the parquet dataset
import pandas as pd
df = pd.read_parquet('/Users/zhouyiying/Downloads/assignment 1 files/prices.parquet')
#convert parquet to DataFrame
df_price = pd.DataFrame(df)
df_price.head(3)
#calculate the percentage of Monday and Tuesday perchase
percentage_of_buying = df_price.groupby("price").agg({
    'mon_purchases': 'mean',
    'tues_purchases': 'mean'
}) 
percentage_of_buying = percentage_of_buying.reset_index()
percentage_of_buying.head(5)

#calculate the percentage of Monday and Tuesday perchase
percentage_of_buying = df_price.groupby("price").agg({
    'mon_purchases': 'mean',
    'tues_purchases': 'mean'
}) 
percentage_of_buying = percentage_of_buying.reset_index()
percentage_of_buying.head(5)

#find the EMV and its price on Tuesday
Tuesday_EMV = percentage_of_buying["Expected Money on Tuesday"].idxmax()
Optimal_Tuesday_Price = percentage_of_buying.loc[Tuesday_EMV]
print("the EMV for Tuesday is:{}".format(Tuesday_EMV))
print("the optimal price is:{}".format(Optimal_Tuesday_Price))
#Expected value on Monday
percentage_of_buying["Expected Money on Monday"] = percentage_of_buying["price"]*percentage_of_buying["mon_purchases"] +(1-percentage_of_buying["mon_purchases"])*Tuesday_EMV
#find the EMV and its price on Monday
Monday_EMV = percentage_of_buying["Expected Money on Monday"].idxmax()
Optimal_Monday_Price = percentage_of_buying.loc[Monday_EMV]
print("the EMV for Tuesday is:{}".format(Monday_EMV))
print("the optimal price is:{}".format(Optimal_Monday_Price))
