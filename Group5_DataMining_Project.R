"BAN620 Data Mining, Project
Group 5 - Abhishek Yadav, Devi Nadimpally, Priyanka Shah
          Sayali Peshwe, Uttara Dabbiru, Victor Antony"

library(dummies)
library(forecast)
library(reshape)
library(leaps)
library(FNN)
library(caret)
library(rpart)
library(rpart.plot)
library(glmulti)
library(gains)
library(neuralnet)
library(FNN)
library(scales)
library(RColorBrewer)
library(ggplot2)
library(dplyr)
library(tidyr)

#Load data
books.main <- read.csv("bestsellers with categories.csv", header = TRUE)  # load data
books.df <-books.main
summary(books.df)
dim(books.df)
#Creating additional column Words_Name that contains the number of words used to name the book
books.df$Words_Name <- sapply(strsplit(books.df$Name, " "), length)

#Merge duplicate authors into one
books.df$Author <- ifelse(books.df$Author=='George R. R. Martin', 'George R.R. Martin', books.df$Author)
books.df$Author <- ifelse(books.df$Author=='J. K. Rowling', 'J.K. Rowling', books.df$Author)
#Creating additional column Popularity that contains levels (Low, Medium, High) based on the number of times an author has appeared in the best sellers list
plot(table(books.df$Author)[books.df$Author])
books.df$count <- as.numeric(table(books.df$Author)[books.df$Author])
books.df$popularity <- ifelse(books.df$count==1,'Low',ifelse(books.df$count==2,'Medium','High'))

#Replaced Books with price=0 as 3.5. Avg of 0 and 1st Quartile value 7.
books.df$Price <- ifelse(books.df$Price==0, mean(books.df$Price), books.df$Price)
books.df[,c(7,10)] <- lapply(books.df[,c(7,10)] , factor)
summary(books.df)
books_new.df <- books.df[,c(-1,-2,-6,-9)]
summary(books_new.df)

round(cor(books_new.df[,c(1,2,3,5)]),2)
#Negative correlation between Words_Name & Reviews
#Negative correlation between Reviews & Price
#Negative correlation between User.Rating & Price
#User Rating and Reviews have a negative correlation with Price.
#Price and Words_Name have a negative correlation with Reviews.

#Heatmap to visualize the correlation
plot(books_new.df[,c(1,2,3,5)])
plot(books_new.df$Reviews~books_new.df$Words_Name)
plot(books_new.df$Reviews~books_new.df$Price)
plot(books_new.df$Price~books_new.df$User.Rating)

##Partitioning the data:
set.seed(1)
test.rows <- sample(c(1:dim(books_new.df)[1]),50)
test.df <- books_new.df[test.rows,]
dim(test.df)
train.rows <- sample(setdiff(rownames(books_new.df), test.rows),dim(books_new.df[-test.rows,])[1]*0.6)
train.df <- books_new.df[train.rows,]
dim(train.df)
valid.rows <- setdiff(rownames(books.df), union(train.rows, test.rows))
valid.df <- books_new.df[valid.rows,]
dim(valid.df)

#==================================================================#
# stacking Price values for each combination of Year and Genre
mlt <- melt(books.main, id=c("Year", "Genre"), measure=c("Price"))
head(mlt, 5)
# use cast() to reshape data and generate pivot table
GenrevsPrice<- cast(mlt, Year ~ Genre, subset=variable=="Price",
      mean)
#plot price trend throughout the years for fiction and non-fiction books
plot(GenrevsPrice$Year, GenrevsPrice$Fiction, type = "b", col = 'Blue',
     ylim = c(0,25), xlim = c(2009,2019),
     xlab = 'Year', ylab='Price')
lines(GenrevsPrice$Year, GenrevsPrice$`Non Fiction`, type = "b", col = 'Red')
legend("topleft", inset=c(0,0),
       legend = c("Fiction","Non Fiction"),
       col = c("blue","red"),
       pch = 1, cex = 0.5)

#Run LM on all variables
books.mvr <- lm(Price ~  ., data=train.df)
options(scipen=999) # avoid scientific notation
summary(books.mvr)
#books.mvr$xlevels$Author <- union(books.mvr$xlevels$Author, levels(booksvalid.df$Author))
books.pred <- predict(books.mvr, newdata = valid.df)
accuracy(books.pred, valid.df$Price)

# Adjusted R sqaure 0.04468, Residual standard error: 9.244
#Important variables
#User.Rating , Genre and popularity

names(books_new.df)
search <- regsubsets(Price ~ ., data =train.df, nbest = 1, nvmax = dim(train.df)[2], method = "exhaustive")
sum <- summary(search)
# show models
sum$which
sum$adjr2
names(train.df)

books.lm.step <- step(books.mvr, direction = "backward")
summary(books.lm.step)  # Which variables were dropped? -- Reviews and Words_Name
books.lm.step.pred <- predict(books.lm.step, valid.df)
accuracy(books.lm.step.pred, valid.df$Price)

#final model
booksfinal.mvr<-lm(Price ~  ., data=train.df[,c(-1,-2,-5)])
options(scipen=999) # avoid scientific notation
summary(booksfinal.mvr)
booksfinal.pred <- predict(booksfinal.mvr, newdata = valid.df)
accuracy(booksfinal.pred, valid.df$Price)
mean(booksfinal.pred)

#==================================================================#
#Predicting reviews

summary(books_new.df)
books2.df <- books_new.df
#Making dummies for genre and popularity:
library(dummies)
Genre <- dummy(books2.df$Genre, sep = "_")
popularity <- dummy(books2.df$popularity, sep = "_")
books2.df <- cbind(books2.df[,c(-4,-6)], Genre,popularity)
names(books2.df)[names(books2.df)=="Genre_Non Fiction"] <- "Genre_Non_Fiction"

summary(books2.df)

#Partitioning the data  into 3 parts:
set.seed(1)
test.rows <- sample(c(1:dim(books2.df)[1]),50)
test.df <- books2.df[test.rows,]
train.rows <- sample(setdiff(rownames(books2.df), test.rows),dim(books2.df[-test.rows,])[1]*0.6)
train.df <- books2.df[train.rows,]
valid.rows <- setdiff(rownames(books2.df), union(train.rows, test.rows))
valid.df <- books2.df[valid.rows,]

#Dimensions of the three data partitions: training, validation and testing.
dim(train.df)
dim(valid.df)
dim(test.df)

# initialize normalized training, validation data, assign (temporarily) data frames to originals
train.norm.df <- train.df
valid.norm.df <- valid.df
books2.norm.df <- books2.df
test.norm.df <- test.df

head(train.norm.df)
summary(train.norm.df)

#Normalize the data for kNN

# use preProcess() from the caret package to normalize the variables:
library(lattice)
library(ggplot2)
library(caret)
norm.values <- preProcess(train.df[,-2], method=c("center", "scale"))
head(norm.values)
train.norm.df <- predict(norm.values, train.df[,-2])
head(train.norm.df)
##Similarly scale valid data,bank2 data and the new data
valid.norm.df <- predict(norm.values, valid.df[,-2])
books2.norm.df <- predict(norm.values, books2.df[,-2])
test.norm.df <- predict(norm.values, test.df)
head(books2.norm.df)
summary(books2.norm.df)

# kNN  regression model to predict the number of reviews (min no of books bought)
# Initial model with k=3
library(FNN)
book_knnreg <- knn.reg(train = train.norm.df, test = valid.norm.df,
                       train.df[, 2], k = 3)

print(book_knnreg)

#Check the accuracy of the kNN regression model through plotting

plot(valid.df[,2], book_knnreg$pred, xlab="Actual no of reviews", ylab="Predicted no of reviews")

#The regression model with k=3 does not look very efficient. We need to calculate the RMSE and MAE values


#RMSE on Validation set:
sqrt(mean((valid.df[,2] - book_knnreg$pred) ^ 2))#9850.182

#Finding a better value of k:
book.accuracy.df <- data.frame(k = seq(1, 20, 1), RMSE = rep(0, 20))
book.accuracy.df

for(i in 1:20) {
  books_knn_chk.pred <- knn.reg(train= train.norm.df, test= valid.norm.df,
                                train.df[,2], k = i)
  book.accuracy.df[i, 2] <- sqrt(mean((valid.df[,2] - books_knn_chk.pred$pred) ^ 2))
  #book.accuracy.df[i, 3] <- mean(abs(valid.df[,2] - books_knn_chk.pred$pred))
}
book.accuracy.df

#Best accuracy is achieved for k=3.As observed from the above developed table.

# Predicting min. number of reviews for new data using the kNN model.

library(FNN)
book_knnreg_prednew <- knn.reg(train = train.norm.df, test = test.norm.df[,-2],
                               train.df[, 2], k = 3)
#Min of reviews for new data
print(book_knnreg_prednew)
mean(book_knnreg_prednew$pred)#13283.24
#RMSE
sqrt(mean((valid.df[,2] - book_knnreg_prednew$pred) ^ 2))#13933.44

#For predicting the no of reviews as earlier we shall be using Regression Trees:

library(rpart)
library(rpart.plot)
library(ISLR)
library(dplyr)
library(tree)
library(tibble)

summary(books2.df)
books3.df<-books_new.df[,]
dim(books3.df)

set.seed(1)
test.rows <- sample(c(1:dim(books3.df)[1]),50)
testtree.df <- books3.df[test.rows,]

train.rows <- sample(setdiff(rownames(books3.df), test.rows),dim(books3.df[-test.rows,])[1]*0.6)
traintree.df <- books3.df[train.rows,]

valid.rows <- setdiff(rownames(books3.df), union(train.rows, test.rows))
validtree.df <- books3.df[valid.rows,]


#Dimensions of the three data partitions: training, validation and testing.
dim(traintree.df)
dim(validtree.df)
dim(testtree.df)

#Generating default Regression tree:
tree.books <- rpart(Reviews ~., data = traintree.df)
options(scipen=999)
summary(tree.books)

#Default Regression Tree structure
rpart.plot(tree.books)

#Importance of different predictors on the outcome variable (Reviews) based on Default Tree:
tree.books$variable.importance

#Using the default Regression Tree for predicting reviews

pred_tree=predict(tree.books,newdata=validtree.df)

#Calculating the RMSE and MAE for the default Regression Tree

#RMSE
sqrt(mean((validtree.df$Reviews-pred_tree)^2)) #RMSE=10036.91

#Pruning the tree based on best cp:
Bookscv.ct <- rpart(Reviews ~ ., data = traintree.df,
                    cp = 0.00001, minsplit = 10,xval=5)
rpart.plot(Bookscv.ct)
printcp(Bookscv.ct)

Bookspruned.ct <- prune(Bookscv.ct,
                        cp = 0.028723290)
length(Bookspruned.ct$frame$var[Bookspruned.ct$frame$var == "<leaf>"])

rpart.plot(Bookspruned.ct)
pred_bestpr=predict(Bookspruned.ct,newdata=validtree.df)

#RMSE for Tree with best cp:
sqrt(mean((validtree.df$Reviews-pred_bestpr)^2))#RMSE=9668.707

#Random Forest
library(randomForest)
## random forest
set.seed(500)
Booksrf <- randomForest(Reviews ~ ., data = traintree.df, ntree = 200,
                        mtry = 4, nodesize = 5, importance = TRUE)

#Plot showing influential variables based on Random Forest:
varImpPlot(Booksrf, type = 1)

pred_rf=predict(Booksrf,newdata=validtree.df)
#RMSE for Random Forest:
sqrt(mean((validtree.df$Reviews-pred_rf)^2)) #RMSE=8457.916

#Optimizing number of trees:

Booksrf.accuracy.df <- data.frame(k = seq(10, 5000, 100), rmse = rep(0, 50))
Booksrf.accuracy.df

for(i in 1:50) {
  set.seed(500)
  Booksrf <- randomForest(Reviews ~ ., data = traintree.df, ntree = i,
                          mtry = 4, nodesize = 5, importance = TRUE)
  #varImpPlot(Booksrf, type = 1)
  pred_opt=predict(Booksrf,newdata=validtree.df)
  Booksrf.accuracy.df[i, 2] <- sqrt(mean((validtree.df$Reviews - pred_opt)^2))
  #Booksrf.accuracy.df[i, 3] <- mean(abs(validtree.df$Reviews - pred_opt))
}

Booksrf.accuracy.df

min(Booksrf.accuracy.df$rmse)#8413.88
#least RMSE is achieved for k=2010

#Randomforest with 710 trees:
library(randomForest)
## random forest
set.seed(500)
Booksrf_best <- randomForest(Reviews ~ ., data = traintree.df, ntree = 2010,
                             mtry = 4, nodesize = 5, importance = TRUE)

#Plot showing influential variables:
varImpPlot(Booksrf_best, type = 1)

pred_rfbest=predict(Booksrf_best,newdata=validtree.df)
#RMSE for optimized Randome Forest:
sqrt(mean((validtree.df$Reviews - pred_rfbest)^2))#8526

#Predicting for test set based on just Random Forest:
testpred_rf=predict(Booksrf,newdata=testtree.df)
#RMSE for Random Forest:
sqrt(mean((testtree.df$Reviews-pred_rf)^2))#11249.4

#Average Prediction on test set:
mean(testpred_rf)#13899.91

##----------------------------------------------------------------------------------
#Models to predict Genre

books.df[,c(7,10)] <- lapply(books.df[,c(7,10)] , factor)

# remove Name variable
books_genre.df <- books.df[,-c(1,2,6,9)]

# pivot table of Year and Genre
library(reshape)
seller_price <- melt(books.df, id=c("Year", "Genre"), measure=c("Price"))
# use cast() to reshape data and generate pivot table (Automatic & Airco)
cast(seller_price, Year ~ Genre, subset=variable=="Price",
     margins=c("grand_row", "grand_col"), mean)
#Fiction books price fluctuates through out the period and non-fiction books price increases 2009 to 2014 and after a 50% drop it started increasing.

# scatterplot price, User.Rating with Genre
library(ggplot2)
theme_set(theme_bw())
ggplot(books.df, aes(x =User.Rating , y = Price)) +
  geom_point(aes(color = factor(Genre)),size = 2) +
  scale_color_brewer(palette = "Dark2")+
  labs(x="User.Rating",y="Price",
       color=factor("Genre"),title="Price Vs User rating")+
  theme(plot.title = element_text(hjust = 0.5))
# There are very few fiction books have less than 4 user rating. Most of books price is under 25 which has more than 4.2 user rating.

##Partition the data into train(60%), valid(40%).
set.seed(1)

train1.rows <- sample(rownames(books_genre.df), dim(books_genre.df)[1]*0.5)
train1.df <- books_genre.df[train.rows, ]

valid1.rows <- sample(setdiff(rownames(books_genre.df), train.rows), dim(books_genre.df)[1]*0.3)
valid1.df <- books_genre.df[valid.rows, ]

test1.rows <- setdiff(rownames(books_genre.df), union(train.rows, valid.rows))
test1.df <- books_genre.df[test.rows, ]

# logistic model to predict Genre
bookslogit.reg <- glm(Genre ~ ., data = train1.df, family = "binomial")
options(scipen=999)
summary(bookslogit.reg)

bookslogit.reg.pred <- predict(bookslogit.reg, valid1.df[,-4], type = "response")

#generate confusion matrix
confusionMatrix(as.factor(ifelse(bookslogit.reg.pred>0.5,'Non Fiction','Fiction')), as.factor(valid1.df[, 4]))

#predict for new data
new.df <- data.frame(User.Rating = 4.6,Reviews=38000,Price=13,Words_Name=7,popularity='Medium')
new.df[,c(5)] <- lapply(new.df[,c(5)] , factor)
new.bookslogit.pred <- predict(bookslogit.reg, new.df, type = "response")
(newdata.odds <-ifelse(new.bookslogit.pred>0.5,'Non Fiction','Fiction'))

#running gains
booksgain <- gains(ifelse(valid1.df$Genre=='Non Fiction',1,0), bookslogit.reg.pred, groups=10)
booksgain

# plot lift chart
plot( c(0,booksgain$cume.pct.of.total*sum(ifelse(valid1.df$Genre=='Non Fiction',1,0)))~c(0,booksgain$cume.obs),
      xlab="# cases", ylab="Cumulative", type="l",
      cex.main=2, cex.lab=1.3, cex.axis=1.7,lwd=2,col = 'Blue')
title(main = "Lift Curve")
par(mar=c(5,7,4,4))
lines(c(0,sum(ifelse(valid1.df$Genre=='Non Fiction',1,0)))~c(0, dim(valid1.df)[1]), lty=2,lwd = 2,col = 'Red',main = "Test")
box(lty = "solid", col = 'black', which = 'outer' , lwd = 15)

# compute deciles and plot decile-wise chart
heights <- booksgain$mean.resp/mean(valid1.df$Genre=='Non Fiction',1,0)
decileplot <- barplot(heights, names.arg = booksgain$depth, ylim = c(0,2),
                      xlab = "Percentile", ylab = "Mean Response/Overall Mean", main = "Decile Chart",cex.main=1.7, cex.lab= 1.7, cex.axis=1.5,lwd=2,col = 'Dark Blue')
par(mar=c(5,7,4,4))

# add labels to columns
text(decileplot, heights+0.2, labels=round(heights, 1), cex = 1.5, pch = 1)
box(lty = "solid", col = 'black', which = 'outer' , lwd = 15)

#CART for Genre

booksdefault.tree <- rpart(Genre ~ ., data = train1.df,
                           control = rpart.control( maxdepth = 7, minbucket = 30),
                           method = "class")
length(booksdefault.tree$frame$var[booksdefault.tree$frame$var == "<leaf>"])
#Plot the default tree as well
prp(booksdefault.tree, type = 1, extra = 4, split.font = 1, varlen = -10)

#predict on validation data
books.pred.valid <- predict(booksdefault.tree,valid1.df[,-4],type = "class")
summary(books.pred.valid)
# generate confusion matrix for validation data
confusionMatrix(books.pred.valid, valid1.df$Genre)


#Prune tree

bookscv.ct <- rpart(Genre ~ ., data = train1.df,
                    cp = 0.00001, minsplit = 10,xval=5)
printcp(bookscv.ct)

bookspruned.ct <- prune(bookscv.ct, cp = 0.03600)
options(scipen = 999)
#Plot the pruned tree
prp(bookspruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col=ifelse(bookspruned.ct$frame$var == "<leaf>", 'gray', 'white'))

booksprun.pred.valid <- predict(bookspruned.ct,valid1.df,type = "class")
confusionMatrix(booksprun.pred.valid, valid1.df$Genre)

## Random Forest
library(randomForest)
booksrf <- randomForest(Genre ~ ., data = train1.df, ntree = 500,
                        mtry = 4, nodesize = 5, importance = TRUE)

## variable importance plot
varImpPlot(booksrf, type = 1)

booksRF.pred.valid <- predict(booksrf,valid1.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(booksRF.pred.valid, valid1.df$Genre)

# test set
booksRF.pred.test<- predict(booksrf,test1.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(booksRF.pred.test, test1.df$Genre)
#################################################################################################################
