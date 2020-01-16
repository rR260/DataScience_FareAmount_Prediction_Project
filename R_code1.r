rm(list=ls())                         
setwd("C:/ProgramData/Microsoft/Windows/Start Menu/Programs")
getwd()
df=read.csv("C:/Users/user/Downloads/DataScience2/train_cab.csv",header=TRUE,na.strings=c("","NA"))
df1=read.csv("C:/Users/user/Downloads/DataScience2/test.csv",header=TRUE)
view(df)
colnames(df)
str(df)
dim(df)
head(df,50)
class(df$fare_amount)
class(df$pickup_datetime)
miss_val=data.frame(apply(df,2,function(x){
  sum(is.na(x))
}))
miss_val1=data.frame(apply(df1,2,function(x){
  sum(is.na(x))
}))
miss_val1
df[1,4]
df[1,4]=NA
df$pickup_latitude[is.na(df$pickup_latitude)]=mean(df$pickup_latitude,na.rm=T)
df[1,4]
df$fare_amount=as.factor(df$fare_amount)
df$fare_amount[is.na(df$fare_amount)]=median(df$fare_amount,na.rm=T)
df[1,4]
require(DMwR)
df=knnImputation(df, k=2)
sum(is.na(df))
class(df)
require(corrplot)
df1$pickup_datetime=as.numeric(df1$pickup_datetime)
df1$pickup_longitude=as.numeric(df1$pickup_longitude)
df1$pickup_latitude=as.numeric(df1$pickup_latitude)
df1$dropoff_longitude=as.numeric(df1$dropoff_longitude)
df1$dropoff_latitude=as.numeric(df1$dropoff_latitude)
df1$passenger_count=as.numeric(df1$passenger_count)
df$pickup_datetime=as.numeric(df$pickup_datetime)
df$pickup_longitude=as.numeric(df$pickup_longitude)
df$pickup_latitude=as.numeric(df$pickup_latitude)
df$dropoff_longitude=as.numeric(df$dropoff_longitude)
df$dropoff_latitude=as.numeric(df$dropoff_latitude)
df$passenger_count=as.numeric(df$passenger_count)
df1$fare_amount=as.numeric(df1$fare_amount)
df$fare_amount=as.numeric(df$fare_amount)
M<-cor(df)                                              
head(round(M,2))
corrplot(M, method="circle")                       
corrplot(M, method="color")                       
corrplot(M, method="number") 
require(rpart)
require(MASS)
require(usdm)
require(tibble)
df1=add_column(df1,fare_amount=df$fare_amount,.before="pickup_datetime")
lm_model=lm(fare_amount~.,data=df)
summary(lm_model)
pred=predict(lm_model,df1[,2:7])
pred
require(DMwR)
regr.eval(df1[,1],pred)



lm_model=lm(fare_amount~.-pickup_datetime,data=df)
summary(lm_model)
confint(lm_model)
pred=predict(lm_model,df1[,2:7])
pred
require(DMwR)
regr.eval(df1[,1],pred)



require(class)
KNN_pred=knn(df,df1,df$fare_amount,k=3) 
KNN_pred
conf_matrix=table(KNN_pred,df1$fare_amount)
conf_matrix
sum(diag(conf_matrix))/nrow(df1)



library(randomForest)
RF=randomForest(fare_amount~.,df,importance=TRUE,ntree=500)
pred=predict(RF,df1[,])


require(tibble)                                  
df1=add_column(df1,fare_amount=pred,.before="pickup_datetime")