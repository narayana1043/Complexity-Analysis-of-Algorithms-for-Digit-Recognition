# setting working dir
setwd("E:\\google_drive\\project_works\\digit-recognition\\using-R")

# training data set 42,000 observations(images)
# column [1] - class Label
# columns [2: ] - pixel data

train <- read.csv(file = "E:\\google_drive\\project_works\\digit-recognition\\data\\train.csv", header = TRUE)
my_matrix <- as.matrix(train[1,-1])
dim(my_matrix) <- c(28,28)
image(my_matrix)

my_matrix1 <- as.matrix(c(0,255,240,0))
dim(my_matrix1) <- c(2,2)
image(my_matrix1)

