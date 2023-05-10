

# # from scipy.stats import norm
# # import seaborn as sns
# # import numpy as np
# # data=np.arange(1,101)
# # data.mean()
# # data.std()
# # pdf=norm.pdf(data,loc=50.5,scale=28.8)
# # sns.lineplot(data,pdf,color='green')
# # cdf=norm.cdf(data,loc=50.5,scale=28.8)
# # sns.lineplot(data,cdf,color='pink',alpha=0.5)
# # cdfn=norm(loc=50,scale=10).cdf(84)
# # prob=1-cdfn


# # val_60=norm(loc=60,scale=15).cdf(60)
# # val_80=norm(loc=60,scale=15).cdf(80)
# # prob=val_80-val_60

# # data_normal=norm.rvs(size=100,loc=50.5,scale=28.8)

# # ax=sns.distplot(data_normal,bins=10,kde=True,color='sky blue',hist_kws={'linewidth':10,'alpha':0.5})
# # ax.set(xlabel='normal distribution',ylabel='frequency')
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom
# pb=binom(n=12,p=0.5)
# x=np.arange(1,13)
# pmf=pb.pmf(x)
# plt.figure(300)
# plt.vlines(x,0,pmf,colors='red',linestyles='dashed',lw=5)
# plt.ylabel("probality")
# plt.xlabel("intervals")
# cdfb=binom(n=12,p=0.25).cdf(8)
# prob=1-cdfb
# cdf4=binom(n=12,p=0.25).cdf(4)
# cdf8=binom(n=12,p=0.25).cdf(8)
# prob=cdf4-cdf8
# data_binom=binom.rvs(n=12,p=0.25,size=10)
# ax=sns.distplot(data_binom,bins=5,kde=True,color='skyblue',hist_kws={"linewidth":5,'alpha':1})
# ax.set(xlabel='binomial distribution',ylabel='frequency') 



# #poisson distribution
# import seaborn as sb
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import poisson
# x=np.arange(1,11)
# pmf=poisson.pmf(x,6)
# plt.vlines(x,0,pmf,colors='k',linestyles='-',lw=6)
# plt.ylabel('probability')
# plt.xlabel('intervals') 

# #poisson cumulative distribution
# #if there are twelve cars crossing a bridge per minute on average ,find the probability of having seventeen or more 
# #cars crossing the bridge in a particular minute.

# cdfp=poisson.cdf(17,mu=12)
# prob=1-cdfp
# cdfp1 =poisson.cdf(17,mu=12)
# cdfpu=poisson.cdf(20,mu=12)
# prof=cdfpu-cdfp1


# data_poisson=poisson.rvs(mu=12,size=10)
# plt.figure(dpi=300)
# ax=sns.distplot(data_poisson,bins=15,kde=True,color='skyblue',hist_kws={'linewidth':16,'alpha':1})
# ax.set(xlabel='poisson distribution',ylabel='frequency')








# #simple linear regression:
# #importing the libraries

# import numpy as np
# import matplotlib.pyplot as ph
# import pandas as pd
# dataset=pd.read_csv("C:/Users\gokul\Downloads\Data Set\Salary_Data.csv")
# x=dataset.iloc[:,0:1].values
# y=dataset.iloc[:,-1].values
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(x_train,y_train)
# regressor.intercept_
# regressor.coef_
# pred_y=26777.391341197632*(10.3)+9360.26128619



# #predicting the test set results.
# y_pred=regressor.predict(x_test)
# regressor.score(x_train,y_train)

# regressor.score(x_test,y_test)  

# plt.figure(dpi=300)
# plt.scatter(x_train,y_train,color='red')
# plt.plot(x_train,regressor.predict(x_train),color='blue')
# plt.title('salary vs experience(training set)') 
# plt.ylabel('years of experiance')
# plt.ylabel('salary')
# plt.show()       
# plt.figure(dpi=300)
# plt.scatter(x_test,y_test,color='blue')
# plt.plot(x_test,regressor.predict(x_test),color='green')
# plt.title('salary vs experience(training set)')
# plt.ylabel('years of experience')
# plt.xlabel('salary') 
# plt.show()    
# from sklearn.metrics import r2_score
# r2_score(y_test, y_test)                                               
# from sklearn.metrics import mean_squared_error
# mean_squared_error(x_test, y_pred)


# np.sqrt(mean_squared_error(y_test,y_pred))
#   from sklearn.metrics import mean_absolute_error
#   mean_absolute_error(y_test, y_pred)
 
#   from sklearn.model_selection import cross_val_score
#   print(cross_val_score(regressor,x,y,cv=25,scoring="explained_variance").mean())
#   ex_var=cross_val_score(regressor,x,y,cv=25,scoring="explained_variance")
 
 
# from mlxtend.evaluate import bias_variance_decomp
#   mse,bias,variance=bias_variance_decomp(regressor),
#         x_train,y_train,x_test,y_test

# num.round= 5,random_speed=1 2 3
# print("average bias:",bias)
# print("averagr variance;", variance)


# # multilinear reression:
# d=pd.read_csv("C:/Users/gokul/Downloads/Data Set/mtcars.csv")
# x=d.iloc[:,[4,6]].values
# y=d.iloc[:,[1]].values
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(x_train,y_train)
# regressor.intercept_
# regressor.coef_
# pred_y=40.0792952+(-0.0304714*95)+(-4.7494281*3.15)
# y_pred=regressor.predict(x_test)
# regressor.score(x_train,y_train)
# regressor.score(x_test,y_test)



# d=pd.read_csv("C:/Users/gokul/Downloads/Data Set/50_Startups.csv")
# x=d.iloc[:,:-1].values
# y=d.iloc[:,-1].values

# import seaborn as sns
# sns.heatmap(d.corr(),annot=True,cmap='RdYlBu',center=0)
# sns.pairplot(d)
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')

# x=np.array(ct.fit_transform(x))
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(x_train,y_train)
# regressor.score(x_train,y_train)
# regressor.intercept_
# regressor.coef_
# y_pred=regressor.predict(x_test)

# # polynomial regression:
# d=pd.read_csv("C:/Users/gokul/Downloads/Data Set/Position_Salaries.csv")
# x=d.iloc[:,1:-1].values
# y=d.iloc[:,-1].values
# ph.scatter(x,y)
# from sklearn.linear_model import LinearRegression
# lin_rge=LinearRegression()
# lin_rge.fit(x,y)
# LinearRegression([6.5])
# from sklearn.linear_model import LinearRegression
# lin_rge=LinearRegression()
# lin_rge.fit(x,y)
# lin_rge.predict([[6.5]])
# ph.scatter(x,y,color='red')
# ph.plot(x,lin_rge.predict(x),color='blue')
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg=PolynomialFeatures(degree=7)
# x_poly=poly_reg.fit_transform(x)
# from sklearn.linear_model import LinearRegression
# lin_rge_2=LinearRegression()
# lin_rge_2.fit(x_poly,y)
# lin_rge_2.predict([[6.5]])
# lin_rge_2.predict(poly_reg.fit_transform([[6.5]]))
# ph.figure(dpi=300)
# ph.scatter(x,y,color='red')
# ph.plot(x,lin_rge_2.predict(poly_reg.fit_transform(x)),color='blue')
# ph.show()



# import numpy as np
# import pandas as pd
# data= pd.read_csv("C:/Users/gokul/Downloads/Data Set/mba.csv")
# confidence=0.95
# n=len(data)
# m=np.mean(data.workex)
# st=np.std(data.workex)
# al=1-(95/100)
# cv=1-(al/2)

# import scipy.stats as stats
# z= stats.norm.ppf(.97500)
# mg=z*(st/np.sqrt(n))
# upperCI=m+mg
# lowerCI=m-mg
# upperCI1=np.mean(data.workex)+stats.norm.ppf(0.975)*(st/np.sqrt(n))
# from sklearn.model_selection import train_test_split
# train,test= train_test_split(data,test_size=0.30,random_state= 0)
# nt=len(train)
# mt=np.mean(train.workex)
# stt=np.std(train.workex)
# from scipy.stats import t
# t=stats.t.ppf(.975,nt-1)
# mg=t*(stt/np.sqrt(n))


# d=pd.read_csv("C:/Users/gokul/Downloads/Data Set/Iris_new.csv")
# x=d.iloc[:,0:4].values
# y=d.iloc[:,-1].values
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.transform(x_test)
# from sklearn.tree import DecisionTreeClassifier
# classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
# classifier.fit(x_train,y_train)
# y_pred=classifier.predict(x_test)
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,y_pred)
# acc=(sum(np.diag(cm)/len(y_test)))
# feature_col=d
# feature_col=feature_col.drop('spectype',axis=1)
# d.columns
# from sklearn.tree import plot_tree
# ph.figure(dpi=500)
# dic_tree=plot_tree(decision_tree=classifier,max_depth=20,feature_names=feature_col.columns,class_names=["setosa","vercicolor","verginica"],filled=False,precision=4,rounded=True,fontsize=4)
                   


# d=pd.read_csv("C:/Users/gokul/Downloads/Data Set/Position_Salaries.csv")
# x=d.iloc[:,1:-1].values
# y=d.iloc[:,-1].values
# from sklearn.tree import DecisionTreeRegressor
# regressor=DecisionTreeRegressor(random_state=0)
# regressor.fit(x,y)
# regressor.predict([[4.7]])
# x_grid=np.arange(min(x),max(x),1)
# x_grid=x_grid.reshape((len(x_grid),1))
# ph.scatter(x,y,color='red')
# ph.plot(x_grid,regressor.predict(x_grid),color='blue')
# ph.title('truth or bluff(Decision tree Regression)')
# ph.xlabel('position level')
# ph.ylabel('salary')



# dataset=pd.read_csv("C:/Users/gokul/Downloads/Data Set/Iris_new.csv")
# x=d.iloc[:,0:4].values
# y=d.iloc[:,0:-1].values
# #splitting the dataset
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.transform(x_test)

# from sklearn.ensembleimport RandomForestClassifier
# classifier=RandomForestClassifier(n_estimators=11,criterion="gini",random_state=0)
# classifier.fit(x_train,y_train)
# y_pred =classifier.predict (x_test)
# from sklearn.metrics import confusion_matrix
# cm= confusion_matrix (y_test,y_pred)
# acc=(sum(np.diag(cm))/len(y_test)

     

     
# import requests
# from bs4 import BeautifulSoup as bs
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# url="https://www.amazon.com/Xiaomi-Unlocked-Worldwide-Charger-Truffle/dp/B09GGD1T6K/ref=sr_1_2?crid=B34A4UQ4UEA3&keywords=mi%2Bphones&qid=1681291970&refinements=p_n_feature_twenty-two_browse-bin%3A23488805011&rnid=23488796011&s=wireless&sprefix=mi%2Bph%2Caps%2C319&sr=1-2&th=1"
# response=requests.get(url)
# soup=bs(response.content,"html.parser")
# review=soup.findAll("span",attrs={"data,hook"."review_body"})
# amazon_reviews=[]
# for i in range(len(reviews))
# amazon_reviews.append(reviews[i].get_text())
# for i in range(len(reviews))
#     amazon_reviews.append(reviews[i].get_text())
# with open(r"amazon_review.txt","w",encoding='utf8') as output
# output.write(str)


import tkinter as tk
from tkinter import*
import pandas as pd

alg=Tk()
alg.geometry("500x600")
alg.configure(bg="#EE4000")
alg.title("KNN")


L1=Label(alg,text="Flowers Classification",font=("rosemary"),bg="#EE4000",fg="white").place(x=40,y=20)
L2=Label(alg,text="sepal length in cm",font=("rosemary"),bg="#F4F4F4",fg="black").place(x=40,y=50)
L3=Label(alg,text="sepal Width in cm",font=("rosemary"),bg= "#F4F4F4",fg="black").place(x=40,y=90)
L4=Label(alg,text="petal length in cm",font=("rosemary"),bg= "#F4F4F4",fg="black").place(x=40,y=130)
L5=Label(alg,text="petal width in cm",font=("rosemary"),bg= "#F4F4F4",fg="black").place(x=40,y=170)
L6=Label(alg,text="Predicted Result",font=("rosemary"),bg= "#F4F4F4",fg="black").place(x=40,y=210)

sl=StringVar()
sw=StringVar()
pl=StringVar()
pw=StringVar()

Entry(alg,text=sl).place(x=250,y=50)
Entry(alg,text=sw).place(x=250,y=90)
Entry(alg,text=pl).place(x=250,y=130)           
Entry(alg,text=pw).place(x=250,y=170)
def model():
    dataset=pd.read_csv("C:/Users/gokul/Downloads/Data Set/Iris_new.csv")
    x=dataset.iloc[:,0:4].values
    y=dataset.iloc[:,[-1]].values

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    from sklearn.neighbors import KNeighborsClassifier
    Classifier=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
    Classifier.fit(x,y)
    sl1=float(sl.get())
    sw1=float(sw.get())
    pl1=float(pl.get())
    pw1=float(pw.get())
    
    
    y_pred=Classifier.predict([[sl1,sw1,pl1,pw1]])
    print(y_pred)
    #print("predicted result is :",dataset.iloc[:,[2,3]].values)
    Label(alg,text=y_pred,font=("rosemary"),bg="#EE4000",fg="white").place(x=250,y=210)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test, y_pred)
    acc=(sum(np.diag(cm))//len(y_test))
    
    
def clear():
    Label(alg,text=" "*30,font=("rosemary"),bg="#EE4000",fg="white").place(x=200,y=210) 
    
Button(alg,text="Prediction",command=model).place(x=20,y=350)    
Button(alg,text="Exit ",command=alg.destroy).place(x=100,y=350)
Button(alg,text="clear ",command=clear).place(x=150,y=350)

alg.resizable(0,0)
alg.mainloop()







   
