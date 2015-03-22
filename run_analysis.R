library(caret)
library(parallel)
library(doParallel)

loadData<-function(){
  #remove variables 
  #rm(list = ls())
  
  #load raw data
  training <<- read.csv("pml-training.csv", header = TRUE)
  test  <<- read.csv('pml-testing.csv')
}

cleanData<-function(){
  #remove columns with over a 90% of not a number
  nasPerColumn<<- apply(training,2,function(x) {sum(is.na(x))});
  training <<- training[,which(nasPerColumn <  nrow(training)*0.9)];
  
  
  #remove near zero variance predictors
  nearZeroColumns <<- nearZeroVar(training, saveMetrics = TRUE)
  training <<- training[, nearZeroColumns$nzv==FALSE]
  
  #remove not relevant columns for classification (x, user_name, raw time stamp 1  and 2, "new_window" and "num_window")
  training<<-training[,7:ncol(training)]
  
  #class into factor
  training$classe <<- factor(training$classe)
}

#split training in 60% training and 40% test
splitdata<-function(){
  trainIndex <<- createDataPartition(y = training$classe, p=0.6,list=FALSE);
  trainingPartition <<- training[trainIndex,];
  testingPartition <<- training[-trainIndex,];
}

fitmodel<-function(){
  
  #random seed
  set.seed(3433)
  #parallel computing for multi-core
  registerDoParallel(makeCluster(detectCores()))
  
  #three models are generated:  random forest   ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model
  
  model_rf <<- train(classe ~ .,  method="rf", data=trainingPartition)    
  model_gbm <<-train(classe ~ ., method = 'gbm', data = trainingPartition)
  model_lda <<-train(classe ~ ., method = 'lda', data = trainingPartition)
  
}


#accuracy info for testingPartition (testing) 
accuracyInfo<-function(){
  
  
  print("Random forest accuracy ")
  rf_accuracy<<- predict(model_rf, testingPartition)
  print(confusionMatrix(rf_accuracy, testingPartition$classe))
  print("")
  print("Boosted trees accuracy ")
  gbm_accuracy<<- predict(model_gbm , testingPartition)
  print(confusionMatrix(gbm_accuracy, testingPartition$classe))
  print("")
  print("Linear discriminant analysis")
  lda_accuracy<<- predict(model_lda , testingPartition)
  print()
  print(confusionMatrix(lda_accuracy, testingPartition$classe))
  
  
}

#tuning the random forest model with CV
CVTuning<-function(){
  
  #random seed
  set.seed(3433)
  #parallel computing for multi-core
  registerDoParallel(makeCluster(detectCores()))  
  controlf <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
  model_rf_CV <<- train(classe ~ ., method="rf",  data=trainingPartition, trControl = controlf)
  print("Random forest accuracy after CV")
  rf_CV_accuracy<<- predict(model_rf_CV , testingPartition)
  print(confusionMatrix(rf_CV_accuracy, testingPartition$classe))
  
}

#get the most important variables in the model
mostImportantVariables<-function(){
print("Variables importance in model")
vi = varImp(model_rf_CV$finalModel)
vi$var<-rownames(vi)
vi = as.data.frame(vi[with(vi, order(vi$Overall, decreasing=TRUE)), ])
rownames(vi) <- NULL
print(vi)
}




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



loadData()
cleanData()
splitdata()
fitmodel()
accuracyInfo()
CVTuning()
mostImportantVariables()
predictionassignmet()

