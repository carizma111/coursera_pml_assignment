# ----------------Fetch data-----------------------------------------------------------------------

data_training <- read.csv("Input\ data/pml-training.csv")
data_testing <- read.csv("Input\ data/pml-testing.csv")

# ----------------Formatting Test data set---------------------------------------------------------
# Remove the columns with NA values from test data set
# It is observed that any column with NA have all row values as NA
# Since these columns are NA for all values, they should not be included in prediction
data_testing <- data_testing[ , colSums(is.na(data_testing)) == 0]

# ----------------Formatting Training data set-----------------------------------------------------
# Select only those features in training data set which are present in test data set, in addition to 
# outcome variable (classe)
data_training_test <- data_training[names(data_training) %in% names(data_testing)]
# Add the outcome variable
data_training_final <- data.frame(cbind(data_training_test, data_training$classe))
# Change the name of the added column to classe
names(data_training_final)[60] <- 'classe'


# ----------------Resolving unbalanced class problem from training data set------------------------
# Remove problem Id [60] from the data_testing
data_testing_new <- data_testing[,-60]

# Remove classe [60] from the data_training_final
data_training_final_new <- data_training_final[,-60]

# Resolving different factor levels issue between training and testing set
data_testing_new$cvtd_timestamp <- as.character(data_testing_new$cvtd_timestamp)
data_testing_new$new_window <- as.character(data_testing_new$new_window)

data_training_final_new$cvtd_timestamp <- as.character(data_training_final_new$cvtd_timestamp)
data_training_final_new$new_window <- as.character(data_training_final_new$new_window)

# Adding isTest identifier
data_testing_new$isTest <- rep(1,nrow(data_testing_new))
data_training_final_new$isTest <- rep(0,nrow(data_training_final_new))

fullSet <- rbind(data_testing_new,data_training_final_new)

fullSet$cvtd_timestamp <- as.factor(fullSet$cvtd_timestamp)
fullSet$new_window <- as.factor(fullSet$new_window)

data_testing_result <- fullSet[fullSet$isTest==1,]
data_training_result <- fullSet[fullSet$isTest==0,]

# Removing isTest identifier
data_testing_result <- data_testing_result[,-60]
data_training_result <- data_training_result[,-60]

# Add the outcome variable
data_training_final_result <- data.frame(cbind(data_training_result, data_training_final$classe))

#Change the name of the added column to classe
names(data_training_final_result)[60] <- 'classe'

# Trainig data is skewed on class A, with more than 1700 observations in A as compared to other classes
# Undersampling -> Randomly delete 1,700 rows containing class 'A' from data_training_final_result
new_subset<- subset(data_training_final_result, classe =='A')

# Randomly select 1700 rows from new_subset
new_subset_sample <- new_subset[sample(nrow(new_subset),1700),]

# Delete rows from data_training_final_result which have rows from new_subset_sample
data_training_final_result_sample <- data_training_final_result[!(data_training_final_result$X %in% new_subset_sample$X),]

# ----------------Remove first column from training and test-----------------------------
data_training_final_result_sample <- data_training_final_result_sample[,-1]
data_testing_result <- data_testing_result[,-1]

# ----------------Importing packages------------------------------------------------------
#install.packages("caret")
library(caret)
#install.packages("e1071")
library(e1071)
#install.packages("randomForest")
library(randomForest)

# ----------------Preparing training and testing set---------------------------------------
# set up training run for x / y syntax because model format performs poorly
x <- data_training_final_result_sample[,-59]
y <- data_training_final_result_sample[,59]

# ----------------Configure parallel processing--------------------------------------------
set.seed(95014)
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# ----------------Configure trainControl object--------------------------------------------
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)


# ----------------Applying random forest----------------------------------------------------
# Applying random forest
modFit_rf <- train(x,y , method="rf", data=data_training_final_result_sample, trControl = fitControl)
print(modFit_rf)

# ----------------De-register parallel processing cluster------------------------------------
stopCluster(cluster)
registerDoSEQ()

# ----------------Predicting on testing set -------------------------------------------------
#Predicting on real test set
prediction_final <- predict(modFit_rf, newdata = data_testing_result)
print(prediction_final)


