# setwd("C:\\Users\\PBSKU\\Box Sync\\MS Data Science\\Self Learning\\Kaggle\\digit-recognition")
setwd("E:/google_drive/project_works/digit-recognition")

install.packages("cape")
library("cape")

# Reading data from files
# 
#   1. train.csv has 785 columns and 42,000 observations. 
#       First column belongs to labels. 
#       Rest 784 columns represent 28x28 px image
# 
#   2. test.csv has 784 clumns and 28,000 observations.
#       All 784 pixels represent 28x28p x images
# 
train_dataset <- read.csv("data\\train.csv", header = TRUE)
test_dataset <- read.csv("data\\test.csv", header = TRUE)

# Extracting image and label data from Training dataset
train_images <- train_dataset[,-1]
train_labels <- train_dataset[,1]
test_images <- test_dataset

# Dimensions of one digit
image_side_length <- 28
image_dim <- c(image_side_length, image_side_length)

# Setting 1px as padding width
padding <- 1

num_images <- 40

banner_side_length <- padding + num_images * (image_side_length + padding)

banner <- matrix(0L, nrow = banner_side_length, ncol = banner_side_length)

# sample_images <- sample(1:length(train_labels), num_images * num_images)


for(i in 0:(num_images-1)) {
  for(j in 1:(num_images-1)) {
    sample_id <- sample(1:length(train_labels), 1)
    temp_image <- as.matrix(train_images[sample_id, ])
    # dim(temp_image) <- image_dim
    temp_image <- matrix(temp_image, nrow = image_side_length, ncol = image_side_length, byrow = TRUE)
    x_axis_range <- (1 + i * (image_side_length + padding) + 1):(1 + i * (image_side_length + padding) + image_side_length)
    y_axis_range <- (1 + j * (image_side_length + padding) + 1):(1 + j * (image_side_length + padding) + image_side_length)
    
    banner[x_axis_range, y_axis_range] <- temp_image    
  }
}

banner = rotate.mat(banner)

image(z = banner)

