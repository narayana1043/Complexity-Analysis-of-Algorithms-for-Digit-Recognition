display_banner <- function(dataset, image_side_length, num_images=40, padding=1) {

  banner_side_length <- padding + num_images * (image_side_length + padding)
  
  banner <- matrix(0L, nrow = banner_side_length, ncol = banner_side_length)
  
  num_observations = dim(dataset)[1]

  for(i in 0:(num_images-1)) {
    for(j in 1:(num_images-1)) {
      sample_id <- sample(1:num_observations, 1)
      
      temp_image <- as.matrix(train_images[sample_id, ])
      temp_image <- matrix(temp_image, nrow = image_side_length, ncol = image_side_length, byrow = TRUE)
      
      x_axis_range <- (1 + i * (image_side_length + padding) + 1):(1 + i * (image_side_length + padding) + image_side_length)
      y_axis_range <- (1 + j * (image_side_length + padding) + 1):(1 + j * (image_side_length + padding) + image_side_length)
      
      banner[x_axis_range, y_axis_range] <- temp_image    
    }
  }
  
  banner = rotate.mat(banner)
  
  image(z = banner)
}
