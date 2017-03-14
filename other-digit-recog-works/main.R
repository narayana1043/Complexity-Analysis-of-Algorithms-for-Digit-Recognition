install.packages("cape")
install.packages("neuralnet")
library("cape")
library("neuralnet")
source("displayData.R")


# Reading data from files
# 
#   1. train.csv has 785 columns and 42,000 observations. 
#       First column belongs to labels. 
#       Rest 784 columns represent 28x28 px image
# 
#   2. test.csv has 784 clumns and 28,000 observations.
#       All 784 pixels represent 28x28p x images
# 
input_dataset <- read.csv("data\\train.csv", header = TRUE)
result_dataset <- read.csv("data\\test.csv", header = TRUE)

# Extracting image and label data from Training dataset
train_images <- input_dataset[1:34000,-1]
cv_images <- input_dataset[34001:38000,-1]
test_images <- input_dataset[38001:42000,-1]


train_labels <- input_dataset[1:34000,1]
cv_labels <- input_dataset[34001:38000,1]
test_labels <- input_dataset[38001:42000,1]

train_labels <- input_dataset[1:34000,-1]
cv_labels <- input_dataset[34001:38000,-1]
test_labels <- input_dataset[38001:42000,-1]



# Dimensions of one digit
image_side_length <- 28
image_dim <- c(image_side_length, image_side_length)

# display_banner(train_images, image_side_length)
# display_banner(test_images, image_side_length)



# Generating Labels for Class 0 through 9
for(i in 0:9) {
  class_label <- paste("Class", i, sep = "")
  train_images <- cbind(train_images, temp_label = (input_dataset$label == i)[1:34000])
  names(train_images)[image_side_length * image_side_length + i + 1] <- class_label
}

# Generating formula
train.labels.as.formula <- paste(names(train_images)[785:794], collapse = "+")
train.features.as.formula <- paste(names(train_images)[1:784], collapse = "+")
f <- as.formula(paste(train.labels.as.formula, train.features.as.formula, sep = " ~ "))

model <- neuralnet(f, data = train_images, hidden = c(14,4))


for(i in 0:9) {
  class_label <- paste("Class ", i, sep = "")
  train_images <- cbind(train_images, class_label = (input_dataset$label == i)[1:34000])
}


head(train_images[, image_side_length * image_side_length + 1 :image_side_length * image_side_length + 10 ], 5)
dim(train_images)


