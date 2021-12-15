df<-read.csv("F:\\Aegis\\Machine Learning\\Topic 3\\Loan-Approval-Prediction.csv")
View(df)
summary(df)
levels(df$Loan_Amount_Term)
str(df)
library(Hmisc)
describe(df)

## part b) Imputation Techniques
## Converting an undefined factor in the categorical variables to NA values
levels(df$Gender)[levels(df$Gender)==""]<-NA
levels(df$Married)[levels(df$Married)==""]<-NA
levels(df$Dependents)[levels(df$Dependents)==""]<-NA
levels(df$Self_Employed)[levels(df$Self_Employed)==""]<-NA

## Checking for outliers
min(boxplot(df$ApplicantIncome)$out)
quantile(df$ApplicantIncome,c(0.1,0.25,0.50,0.75,0.9,0.95,0.97,0.99))
df$ApplicantIncome<-ifelse(df$ApplicantIncome>30000,30000,df$ApplicantIncome)

boxplot(df$CoapplicantIncome)$out
quantile(df$CoapplicantIncome,c(0.50,0.75,0.90,0.99))
quantile(boxplot(df$CoapplicantIncome)$out,c(0.50,0.75,0.9))
df$CoapplicantIncome<-ifelse(df$CoapplicantIncome>11300,11300,df$CoapplicantIncome)
summary(df)

## Filling missing NAs for the Credit History Variable
df$Credit_History=(ifelse(is.na(df$Credit_History)&df$Loan_Status=="Y",1,df$Credit_History))
df$Credit_History=(ifelse(is.na(df$Credit_History)&df$Loan_Status=="N",0,df$Credit_History))
summary(df$Credit_History)

## Converting categorical numerical variables to categorical variables
df$Credit_History<-as.factor(df$Credit_History)
df$Loan_Amount_Term<-as.factor(df$Loan_Amount_Term)
summary(df)
corrgram(df)
## Applicant Income is highly correlated with loan amount

hist(df$ApplicantIncome,breaks=50)
hist(df$CoapplicantIncome,breaks = 50)
boxplot(df$LoanAmount)
plot(df$ApplicantIncome,df$LoanAmount)
plot(df$CoapplicantIncome,df$LoanAmount)
summary(df[df$CoapplicantIncome==0,"ApplicantIncome"])

## Imputing the remaining NA values using MICE
library(mice)
tempdata<-mice(df,method = 'rf')
df_imp<-complete(tempdata)
summary(df_imp)

## part c) Information Value Summary Plot 
## to visualize the strength of the variables
library(devtools)
library(woe)
iv <- iv.mult(df_imp[,names(df_imp)],y="Loan_Status",summary= TRUE)
iv
iv.plot.summary(iv)

## Part d) Dividing data into training data and test data
## Since our data is more biased with with loan status = Y, so we generate unbiased train and test data

# Create Training Data
input_ones <- df_imp[df_imp$Loan_Status == "Y", ]  # all Y's
input_zeros <- df_imp[df_imp$Loan_Status == "N", ]  # all N's
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))  # Y's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_zeros))  # N's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
train<- rbind(training_ones, training_zeros)  # row bind the Y's and N's 

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
test<- rbind(test_ones, test_zeros)  # row bind the Y's and N's 

## Selecting only the relevant independent variables, predictors
##based on the Information values for both the train and test data,
##and also from significant variables(p-values) from the initial model 
View(train)
colnames(train)
train_final<-train[,c(3,4,5,7,8,9,11,12,13)]
test_final<-test[,c(3,4,5,7,8,9,11,12,13)]

## Part e) Building the logistic model
## Running the model
model<-glm(Loan_Status~.,data = train_final,family = "binomial")
summary(model)

results<-predict(model,test_final,type = "response")

## Part f) Optimum Cutoff Probability to reduce missclassification error
table(test_final$Loan_Status,results>0.05)  ## checking for the optimum cutoff

## Using information Value package to calculate the optimal cutoff
library(InformationValue)
optCutOff <- optimalCutoff(test_final$Loan_Status, results)[1] 
optCutOff
## Optimal Cutoff value, threshold is calculated as 0.0317
table(test_final$Loan_Status,results>0.0317)

## To check for missclassification error
library(car)
vif(model)
misClassError(test_final$Loan_Status, results, threshold = optCutOff)

