#Packages 
library(caTools)
library(e1071)
library(caret)
library(randomForest)
library(RColorBrewer)
library(rattle)
Heart=read.csv(choose.files())
str(Heart)
dim(Heart)
Heart$sex=as.numeric(Heart$sex)
Heart$DEATH_EVENT=as.factor(Heart$DEATH_EVENT)


#Spliting the data

split = sample.split(Heart,SplitRatio = 0.8)
train=subset(Heart,split=="TRUE")
test=subset(Heart,split=="FALSE")
x=subset(Heart,select =-DEATH_EVENT)
y=Heart$DEATH_EVENT

#logit model
logit_model=glm(DEATH_EVENT~.,family=binomial(link='logit'),data=train)
summary(logit_model)

anova(logit_model,test= 'Chisq')
fitted_result=predict(logit_model,newdata = test)
test$fitted_result=ifelse(fitted_result>0.5,1,0)
table(test$fitted_result,test$DEATH_EVENT)
confusionMatrix(table(test$fitted_result,test$DEATH_EVENT))

install.packages("RORC")
library(ROCR)
ROCRpred = prediction(test$fitted_result,test$DEATH_EVENT)
ROCRpref= performance(ROCRpred,measure = 'tpr',x.measure = 'fpr')
plot(ROCRpref,colorize='TRUE',print.cutoffs.at=seq(0.1,by=0.1))
abline(a=0,b=1)

auc= performance(ROCRpred,measure = "auc")
auc=auc@y.values[[1]]
legend(.6,.4,auc,title="AUC",cex=1)
auc

#SVM model
svm_model=svm(DEATH_EVENT ~ ., data=train)
summary(svm_model )

test$DEATH_EVENT_predicted=predict(svm_model,newdata = test)
cm=table(test$DEATH_EVENT,test$DEATH_EVENT_predicted)
confusionMatrix(cm)

svm_tune = tune(svm,train.x=x,train.y = y,
                kernel="radial",ranges = list(cost=10^(-1:2),gamma=c(0.5,1,2)))
 

#NAIVE BAYES
naivebayes_model = naiveBayes(DEATH_EVENT ~ ., data=train)
y_predict=predict(naivebayes_model,newdata =test)
cm1=table(test$DEATH_EVENT,y_predict)
confusionMatrix(cm1)
str(train)
summary(naivebayes_model)


#DECISION TREE
library(rpart)
library(rpart.plot)
rpart.plot(DTM)
#DTM2 = rpart(DEATH_EVENT ~ .,data = train,method="class",control=rpart.control(minsplit=2,cp=0))
#fancyRpartPlot(DTM2)
DTM = rpart(DEATH_EVENT ~ .,data = train,method="class")
test$DTMpred= predict(DTM,newdata =test,type ="class" )
confusionMatrix(table(test$DEATH_EVENT,test$DTMpred))
rpart.plot(DTM)
  printcp(DTM)
plotcp(DTM)

min(DTM$cptable[,'xerror'])
which.min(DTM$cptable[,'xerror'])
cpmin=DTM$cptable[2,'xerror']

#After pruning the Decision tree 

DTM_pruned=prune(DTM,cp=cpmin)
rpart.plot(DTM_pruned)

table(test$DEATH_EVENT,DTMpred)
test$DEATH_EVENT_pred=predict(DTM_pruned,newdata = test,type="class")
cm2=table(test$DEATH_EVENT,test$DEATH_EVENT_pred)
confusionMatrix(cm2)


#RANDOM FOREST
set.seed(123)
RF=randomForest(DEATH_EVENT ~ .,data = train,ntree=1000)
test$rfpredict=predict(RF,newdata = test)
cm3=table(test$rfpredict,test$DEATH_EVENT)
confusionMatrix(cm3)
plot(RF)
importance(RF)
varImpPlot(RF)

#Principal Component Analysis

pca=preProcess(x=Heart[-13],method='pca',thresh = 0.9,verbose=TRUE)
names(pca) 
pca$rotation
pca
dataset=predict(pca,Heart)
dim(dataset)

classifier=svm(DEATH_EVENT ~ .,data = dataset)
y.predict=predict(classifier,newdata = dataset[-1])
cm=table(dataset[,1],y.predict)
confusionMatrix(cm)
