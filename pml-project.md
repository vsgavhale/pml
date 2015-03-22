Practical Machine Learning Course Project
========================================================

Introduction
-------------------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement  a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is  to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here:  [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har).


Loading data
-------------------------------


The training data for this project are available   [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the tests data [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) 

After downloading into the working directory, the following code loads them in *training* and *test*. Needed libraries are also loaded.


```r
library(caret)
library(parallel)
library(doParallel)
#load raw data
training <- read.csv("pml-training.csv", header = TRUE)
test  <- read.csv('pml-testing.csv')
```



Cleaning data 
-------------------------------
After studying the summary of the training data, a number of cleaning actions are undertaken:
- Removing columns with over a 90% of not a number
- Removing near zero variance predictors
- remove not relevant columns for classification (x, user name, raw time stamp 1 and 2, new window and num window).
- Convert class into factor


```r
#remove columns with over a 90% of not a number
nasPerColumn<- apply(training,2,function(x) {sum(is.na(x))});
training <- training[,which(nasPerColumn <  nrow(training)*0.9)];  
  
#remove near zero variance predictors
nearZeroColumns <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nearZeroColumns$nzv==FALSE]
  
#remove not relevant columns for classification (x, user_name, raw time stamp 1  and 2, "new_window" and "num_window")
training<-training[,7:ncol(training)]
  
#class into factor
training$classe <- factor(training$classe)
```


Splitting data
-------------------------------
Split the data: 60% for training, 40% for testing


```r
trainIndex <- createDataPartition(y = training$classe, p=0.6,list=FALSE);
trainingPartition <- training[trainIndex,];
testingPartition <- training[-trainIndex,];
```

Create machine learning models
-------------------------------
Three models are generated:  random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. Parallel computing methods are employed to improve efficiency. The Principal Component Analysis (PCA) can also be added as a preprocess option of the train function, but at the expense of losing accuracy.


```r
#random seed
set.seed(3433)
#parallel computing for multi-core
registerDoParallel(makeCluster(detectCores()))
#three models are generated:  random forest   ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model   
model_rf <- train(classe ~ .,  method="rf", data=trainingPartition)    
model_gbm <-train(classe ~ ., method = 'gbm', data = trainingPartition)
model_lda <-train(classe ~ ., method = 'lda', data = trainingPartition) 
```


Studying accuracy
-------------------------------
After using the testing data to make predictions with the models calculated and showing the confusion matrix for each model; the accuracy returned for random forest, boosted trees and linear discriminant analysis are 99%, 96% and 69\%. Therefore, the random forest model is the most promising model to be optimized with cross validation.  This was the expected output since random forest was the most complex technique evaluated because it combines a number of classifiers. 


```r
print("Random forest accuracy ")
rf_accuracy<- predict(model_rf, testingPartition)
print(confusionMatrix(rf_accuracy, testingPartition$classe))
print("Boosted trees accuracy ")
gbm_accuracy<- predict(model_gbm , testingPartition)
print(confusionMatrix(gbm_accuracy, testingPartition$classe))
print("Linear discriminant analysis")
lda_accuracy<- predict(model_lda , testingPartition)
print(confusionMatrix(lda_accuracy, testingPartition$classe))
```

Tuning with cross validation
-------------------------------
To improve the model obtained and more specifically to avoid over-fitting, the cross validation technique is employed with 10 folds.


```r
#random seed
set.seed(3433)
#parallel computing for multi-core
registerDoParallel(makeCluster(detectCores()))  
controlf <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
model_rf_CV <- train(classe ~ ., method="rf",  data=trainingPartition, trControl = controlf)
```



The final accuracy of the model with cross validation is also obtained:


```r
print("Random forest accuracy after CV")
rf_CV_accuracy<<- predict(model_rf_CV , testingPartition)
print(confusionMatrix(rf_CV_accuracy, testingPartition$classe))
```

This accuracy is 0.9921, which is only slightly better than the baseline accuracy (0.9916) which was already very high 


Variables importance
-------------------------------
Given the final model, the importance of the variables in this model can be studied. The following code concludes that *roll_belt* is the most important predictor for the model obtained with random forest and tuned by cross validation.



```r
print("Variables importance in model")
vi = varImp(model_rf_CV$finalModel)
vi$var<-rownames(vi)
vi = as.data.frame(vi[with(vi, order(vi$Overall, decreasing=TRUE)), ])
rownames(vi) <- NULL
print(vi)
```



Predicting 20 test cases
-------------------------------
Finally, the random forest model tuned with cross validation (model_rf_CV) is used to predict  20 test cases available in the test data loaded at the beginning of the project.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


#Prediction Assignment Submission
predictionassignmet<- function(){
  prediction <- predict(model_rf_CV, test)
  print(prediction)
  answers <- as.vector(prediction)
  pml_write_files(answers)
}
```


