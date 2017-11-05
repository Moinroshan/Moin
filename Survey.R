library(dplyr) # Data manuplation
library(ggplot2) # Charts
library(caret) # Data Partition
library(unbalanced) # To handle unbalanced data
library(VIM) # For KNN Imputation
library(randomForest) # For random forest
library(tree) # For Decession Tree

# Importing Data

SSD_Full <- read.csv("C:/Users/his/Desktop/Both/SimulationSurveyData H_L.csv",header = T)
head(SSD_Full)
str(SSD_Full)

# Changing to categorical 

SSD_Full <- SSD_Full %>% mutate(Q1= as.factor(Q1),Q2= as.factor(Q2),Q3= as.factor(Q3),
                                Q4= as.factor(Q4),Q5= as.factor(Q5),Q6= as.factor(Q6),
                                Q7= as.factor(Q7),Q7.1= as.factor(Q7.1),
                                UsingBETA=as.factor(UsingBETA),Risk.Type=as.factor(Risk.Type))
# Replacing Values
SSD_Full$Q1 <- recode(SSD_Full$Q1,"I am responsible for the decision to work with AlPHABETAGAMMA" =1,
                      "I am involved in the day-to-day operational use and/or benefit from the products and services that AlPHABETAGAMMA provides my company"=2,
                      "I am mainly responsible for contract matters with AlPHABETAGAMMA."=3,
                      "I contribute to decisions with regards to purchasing from AlPHABETAGAMMA but do not directly deploy AlPHABETAGAMMA products or services as part of my job."=4)

SSD_Full$Q2 <- recode(SSD_Full$Q2, "Yes"=1, "No"=0 )
SSD_Full$Q3 <- recode(SSD_Full$Q3,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$Q4 <- recode(SSD_Full$Q4,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$Q5 <- recode(SSD_Full$Q5,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$Q6 <- recode(SSD_Full$Q6,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$Q7 <- recode(SSD_Full$Q7,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$Q7.1 <- recode(SSD_Full$Q7.1,"Somewhat Agree" = 1,"Agree"=2,"Strongly Agree"=3,
                      "Neither Agree or Disagree"=4,"Somewhat Disagree"=5,"Disagree"=6,
                      "Strongly Disagree"=7)
SSD_Full$UsingBETA <- recode(SSD_Full$UsingBETA, "TRUE"=1, "FALSE"=0 )

str(SSD_Full)

# Visulaising Cetagorical variable

par(mfrow=c(2,2),cex=0.7, mai=c(0.4,0.3,0.2,0.1))
hist(SSD_Full$Q1)
hist(SSD_Full$Q2)
hist(SSD_Full$Q3)
hist(SSD_Full$Q4)
hist(SSD_Full$Q5)
hist(SSD_Full$Q6)
hist(SSD_Full$Q7)
hist(SSD_Full$Q7.1)

# Setting graphical setting to default
dev.off()

# Detecting outliers

dotchart(SSD_Full$Period,main = "Period")
dotchart(SSD_Full$FEE,main = "FEE")
dotchart(SSD_Full$EXlast30,main = "EXlast30")
dotchart(SSD_Full$DOLast30,main = "DOLast30")
dotchart(SSD_Full$EOLAst30,main = "EOLAst30")
dotchart(SSD_Full$SOLast30,main = "SOLast30")
dotchart(SSD_Full$FMPS,main = "FMPS")

# Remove outliers

summary(SSD_Full$FEE)
SSD_Full$FEE[SSD_Full$FEE > 140] <-NA

summary(SSD_Full$EXlast30)
SSD_Full$EXlast30[SSD_Full$EXlast30 > 330] <- NA

summary(SSD_Full$DOLast30)
SSD_Full$DOLast30[SSD_Full$DOLast30 > 60 ] <- NA

summary(SSD_Full$EOLAst30)
SSD_Full$EOLAst30[SSD_Full$EOLAst30 > 200] <-NA

summary(SSD_Full$SOLast30)
summary(SSD_Full$FMPS)
summary(SSD_Full)

# Treating Missing values by Knn imputation

set.seed(111)

Imp_SSD_Full <- VIM::kNN(SSD_Full,
                           variable = c("FEE","EXlast30","DOLast30","EOLAst30"))
summary(Imp_SSD_Full)

Imp_2_SSD_full <- select(Imp_SSD_Full,c(1:17))

head(Imp_2_SSD_full)
str(Imp_2_SSD_full)

# Splitting Data

splitdata <- createDataPartition(Imp_2_SSD_full$Risk.Type,
                                 p=.7,list = FALSE,times = 1)
train_Imp <- Imp_2_SSD_full[ splitdata, ]
test_Imp <- Imp_2_SSD_full[ -splitdata,]
dim(train_Imp)
dim(test_Imp) 

prop.table(table(train_Imp$Risk.Type))
prop.table(table(test_Imp$Risk.Type))

##########################################
### Create Model using Random Forest #####
##########################################

fit_rf <- randomForest(Risk.Type ~., data = train_Imp)
summary(fit_rf)

# Validating the model
confusionMatrix(predict(fit_rf,test_Imp),test_Imp$Risk.Type)

# Plot AUC curve                      
library(pROC)
aucval.RF <- auc(predictor = as.ordered(predict(fit_rf,data = train_Imp)),
              response = train_Imp$Risk.Type,plot  = TRUE)                    
                                                          
print(aucval.RF) 

# Important Variables
varImpPlot(fit_rf)

#######################################
####### Logistic regression ###########
#######################################
## Difficult to interpretate

fit_log <- glm(Risk.Type ~., data = train_Imp,family = "binomial")
fit_log.pred <- predict(fit_log,type = "response",newdata = test_Imp)

table(fit_log.pred,test_Imp$Risk.Type)
mean(fit_log.pred == test_Imp$Risk.Type)
mean(fit_log.pred != test_Imp$Risk.Type)
summary(fit_log)

confusionMatrix(train_Imp$Risk.Type,predict(fit_log,type = "response",newdata =test_Imp))

######################################
#########    SVM      ################
######################################
svm.fit<- svm(Risk.Type~.,data = train_Imp, kernel= "linear", cost = 0.1, scale = FALSE)
summary(svm.fit)  
confusionMatrix(predict(svm.fit,test_Imp),test_Imp$Risk.Type)

aucval.SVM <- auc(predictor = as.ordered(predict(svm.fit,data = train_Imp)),
                 response = train_Imp$Risk.Type,plot  = TRUE)  
print(aucval.SVM)

svm.fit$index # This are the support Vectors

#Lets Tune Cost

tune.out<-tune(svm,Risk.Type~.,data = train_Imp, kernel = "linear", ranges = 
                 list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out) # The best parameter is Cost 0.1

######################################
######### Decession Tree #############
######################################

tree.fit <- tree(Risk.Type ~., data = train_Imp)
summary(tree.fit)
plot(tree.fit)
text(tree.fit,pretty = 0)

confusionMatrix(predict(tree.fit,test_Imp,type = "class"),test_Imp$Risk.Type)

mean(predict(tree.fit,test_Imp,type = "class")!=test_Imp$Risk.Type)
# So the error is 0.42 % 
# Lets try to prune the Tree

cv_tree<-cv.tree(tree.fit,FUN = prune.misclass)
names(cv_tree)
cv_tree
# Plot size and error(devation)
plot(cv_tree$size,cv_tree$dev,type = "b")
#Now from the graph and cv_tree result we can chose size 9 is best

Pruned_model<-prune.misclass(tree.fit,best = 8)
plot(Pruned_model)
text(Pruned_model,pretty = 0)

confusionMatrix(predict(Pruned_model,test_Imp,type = "class"),test_Imp$Risk.Type)
mean(predict(Pruned_model,test_Imp,type = "class")!=test_Imp$Risk.Type)
# Now error is slightly decressed


###############################################
############ Lets balance the data ############
###############################################

summary(Imp_2_SSD_full$Risk.Type)

library(ROSE)
library(unbalanced)

# Over balancing data

Over <- ovun.sample(Risk.Type ~.,
                    data =train_Imp,method = "over", N =  2300)$data

# Model for balanced dataset

rfover <- randomForest(Risk.Type ~.,data = Over)

confusionMatrix(predict(rfover,test_Imp),test_Imp$Risk.Type)

# By doing over-sampling we have improved sensitivity

# Under sampling

Under <- ovun.sample(Risk.Type ~.,
                    data =train_Imp,method = "under", N =  310)$data

# Model for balanced dataset

rfunder <- randomForest(Risk.Type ~.,data = Under)

confusionMatrix(predict(rfunder,test_Imp),test_Imp$Risk.Type)

# By doing Under sampling we dropped accuracy and sensitivity

# Both Under and Over sampling

both <- ovun.sample(Risk.Type ~.,
                     data =train_Imp,method = "both",p=0.5
                    ,seed = 222, N= 1743)$data
summary(both$Risk.Type)

# Model for balanced dataset

rfboth <- randomForest(Risk.Type ~.,data = both)

confusionMatrix(predict(rfboth,test_Imp),test_Imp$Risk.Type)

## Balancing by using ROSE (Synthetic Data)

balrose <- ROSE(Risk.Type ~.,data =train_Imp,seed = 222, N= 1500)$data

summary(balrose)
summary(Imp_2_SSD_full)
str(Imp_2_SSD_full)
rfrose <- randomForest(Risk.Type ~.,data = balrose)

confusionMatrix(predict(rfrose,test_Imp),test_Imp$Risk.Type)

#################################
# SVM for balanced data

svm.over<- svm(Risk.Type~.,data = Over, kernel= "linear", cost = 0.1, scale = FALSE)
summary(svm.over)  
confusionMatrix(predict(svm.over,test_Imp),test_Imp$Risk.Type)

svm.under<- svm(Risk.Type~.,data = Under, kernel= "linear", cost = 0.1, scale = FALSE)
confusionMatrix(predict(svm.under,test_Imp),test_Imp$Risk.Type)

svm.both <- svm(Risk.Type~.,data = both, kernel= "linear", cost = 0.1, scale = FALSE)
confusionMatrix(predict(svm.both,test_Imp),test_Imp$Risk.Type)

svm.rose <- svm(Risk.Type~.,data = balrose, kernel= "linear", cost = 0.1, scale = FALSE)
confusionMatrix(predict(svm.rose,test_Imp),test_Imp$Risk.Type)

################################
# Decession Tree

tree.over <- tree(Risk.Type ~., data = Over)
confusionMatrix(predict(tree.over,test_Imp,type = "class"),test_Imp$Risk.Type)

tree.under <- tree(Risk.Type ~., data = Under)
confusionMatrix(predict(tree.under,test_Imp,type = "class"),test_Imp$Risk.Type)

tree.both <- tree(Risk.Type ~., data = both)
confusionMatrix(predict(tree.both,test_Imp,type = "class"),test_Imp$Risk.Type)

tree.rose <- tree(Risk.Type ~., data = balrose)
confusionMatrix(predict(tree.rose,test_Imp,type = "class"),test_Imp$Risk.Type)
