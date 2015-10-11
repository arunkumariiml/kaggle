setwd("~/Downloads/kaggle")
library(xgboost)
library(data.table)

set.seed(1)
# Read train, test & store data 

train <- fread("train.csv",stringsAsFactors = T)
test <- fread("test.csv",stringsAsFactors = T)
store <- fread("store.csv",stringsAsFactors = T)

train<-as.data.frame(train)
test <- as.data.frame(test)
store <- as.data.frame(store)

#merge data from store.csv to both train and test
train <- merge(train,store,by="Store")
test <- merge(test,store,by="Store")

cat("train data column names and details\n")
summary(train)
cat("test data column names and details\n")
summary(test)

trainExS <- train
trainExS$Sales <- NULL
trainExS$Customers <- NULL
testExID <- test
testExID$Id <- NULL

combined <- rbind(trainExS,testExID)

# Treat NA values with 0

combined[is.na(combined)] <- 0

# Mark all factor variables appropriately

combined$Store <- factor(combined$Store)
combined$DayOfWeek <- factor(combined$DayOfWeek)
combined$Open <- factor(combined$Open)
combined$Promo <- factor(combined$Promo)
combined$StateHoliday <- factor(combined$StateHoliday)
combined$SchoolHoliday <- factor(combined$SchoolHoliday)
combined$StoreType <- factor(combined$StoreType)
combined$Assortment <- factor(combined$Assortment)
combined$Promo2 <- factor(combined$Promo2)
combined$PromoInterval <- factor(combined$PromoInterval)
combined$Promo2SinceYear <- factor(combined$Promo2SinceYear)
combined$CompetitionOpenSinceYear <- factor(combined$CompetitionOpenSinceYear)
combined$CompetitionOpenSinceMonth <- factor(combined$CompetitionOpenSinceMonth)

# seperating out the elements of the date column for the train set
combined$Date  <- as.Date(combined$Date)
combined$month <- as.integer(format(combined$Date, "%m"))
combined$year  <- as.integer(format(combined$Date, "%y"))
combined$date  <- as.integer(format(combined$Date, "%d"))

combined$Date <- NULL

combined$date <- factor(combined$date)
combined$month <- factor(combined$month)
combined$year <- factor(combined$year)

trainP<-head(combined,nrow(train))
testP<-tail(combined,nrow(test))
trainP$Sales<-train$Sales
feature.names <- names(trainP)[1:(ncol(trainP))-1]

for (f in feature.names){
  if (class(train[[f]])=="factor") {
    levels = unique(c(trainP[[f]],testP[[f]]))
    levels(testP[,f]) = levels(trainP[,f])
  }
}

################ XGB MODELLING #########################

cat(".......Training XGBOOST.........\n")

param <- list("objective" = "reg:linear",    # multiclass classification 
#             "num_class" = 2,    # number of classes 
              "feval" = "RMPSE",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 21,    # maximum depth of tree 
              "eta" = 0.5,    # step size shrinkage 
              "gamma" = 4.5,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree
#             "min_child_weight" = 1,  # minimum sum of instance weight needed in a child 
              "colsample_bytree" = 0.2  # subsample ratio of columns when constructing each tree 
)


# k-fold cross validation, with timing

nround.cv = 800

system.time( xgb_check <- xgb.cv(param=param, data=data.matrix(trainP[,feature.names]), label=trainP$Sales, 
                                 nfold=4, nrounds=nround.cv, missing=NaN,prediction=TRUE, verbose=TRUE) )

min_rmse_idx <- which.min(xgb_check$dt[,test.rmse.mean])

system.time( xgb_chk <- xgboost(param=param, data=data.matrix(trainP[,feature.names]), label=trainP$Sales, 
                                missing=NaN,nrounds=min_rmse_idx, verbose=TRUE) )

gc()

# Run the model using test data set

xgb_val <- data.frame(Id=test$Id)
xgb_val$Sales <- NA
for(rows in split(1:nrow(testP), ceiling((1:nrow(testP))/1000))) {
  xgb_val[rows, "Sales"] <- predict(xgb_chk, data.matrix(testP[rows,feature.names]),missing=NaN)
}

gc()

################ SIMPLE ENSEMBLE #########################

write.csv(xgb_val, "sub_xgb_1.csv", row.names=FALSE)
