# Rosenbrock Function Optimization using Gradient Descent

# Define the Rosenbrock function
rosenbrock <- function(x, y) {
  return((1-x)^2 + 100*(y-x^2)^2)
}

# Define the gradient of the Rosenbrock function
rosenbrock_gradient <- function(x, y) {
  dx <- -2*(1-x) - 400*x*(y-x^2)
  dy <- 200*(y-x^2)
  return(c(dx, dy))
}

# Gradient descent algorithm
gradient_descent <- function(initial_point, learning_rate, epsilon, max_iterations) {
  # Initialize
  x <- initial_point[1]
  y <- initial_point[2]
  
  # Store history for visualization
  history_x <- x
  history_y <- y
  history_loss <- rosenbrock(x, y)
  
  # Iteration counter
  iteration <- 0
  
  # Perform gradient descent
  repeat {
    # Calculate gradient
    grad <- rosenbrock_gradient(x, y)
    
    # Calculate gradient norm
    grad_norm <- sqrt(sum(grad^2))
    
    # Check stopping condition
    if (grad_norm < epsilon || iteration >= max_iterations) {
      break
    }
    
    # Update parameters
    x <- x - learning_rate * grad[1]
    y <- y - learning_rate * grad[2]
    
    # Update counters and history
    iteration <- iteration + 1
    history_x <- c(history_x, x)
    history_y <- c(history_y, y)
    history_loss <- c(history_loss, rosenbrock(x, y))
    
    # Print progress every 1000 iterations
    if (iteration %% 1000 == 0) {
      cat("Iteration:", iteration, "Loss:", rosenbrock(x, y), "Gradient Norm:", grad_norm, "\n")
    }
  }
  
  cat("Final position: x =", x, "y =", y, "\n")
  cat("Final loss:", rosenbrock(x, y), "\n")
  cat("Total iterations:", iteration, "\n")
  
  return(list(
    x = x,
    y = y,
    loss = rosenbrock(x, y),
    iterations = iteration,
    history_x = history_x,
    history_y = history_y,
    history_loss = history_loss
  ))
}

# Set parameters from the problem description
initial_point <- c(-1, 1)
learning_rate <- 0.001
epsilon <- 1e-6
max_iterations <- 100000

# Run gradient descent
result <- gradient_descent(initial_point, learning_rate, epsilon, max_iterations)

# Plot loss vs iterations
plot(1:length(result$history_loss), result$history_loss, 
     type = "l", 
     xlab = "Iterations", 
     ylab = "Loss", 
     main = "Rosenbrock Function Optimization",
     col = "blue")

# Create 2D visualization of the descent path
# First, create a grid for the contour plot
x_range <- seq(-2, 2, length.out = 100)
y_range <- seq(-1, 3, length.out = 100)
z_matrix <- matrix(0, length(x_range), length(y_range))

for (i in 1:length(x_range)) {
  for (j in 1:length(y_range)) {
    z_matrix[i, j] <- rosenbrock(x_range[i], y_range[j])
  }
}

# Plot the contour with the path
contour(x_range, y_range, z_matrix, nlevels = 20, xlab = "x", ylab = "y", 
        main = "Gradient Descent Path", drawlabels = FALSE)
lines(result$history_x, result$history_y, col = "red", lwd = 2)
points(result$history_x[1], result$history_y[1], col = "green", pch = 19, cex = 1.5)
points(result$history_x[length(result$history_x)], 
       result$history_y[length(result$history_y)], 
       col = "blue", pch = 19, cex = 1.5)
legend("topright", 
       legend = c("Path", "Start", "End"), 
       col = c("red", "green", "blue"), 
       pch = c(NA, 19, 19), 
       lwd = c(2, NA, NA), 
       cex = 0.8)

# 3D visualization using base R (persp) - no additional packages needed
# Create a separate figure for the 3D plot
par(mar = c(4, 4, 2, 2))  # Set smaller margins

# Create data for 3D surface - use the same data we already have
plot_resolution <- 50  # Lower resolution for better performance
x_surf <- seq(-2, 2, length.out = plot_resolution)
y_surf <- seq(-1, 3, length.out = plot_resolution)
z_surf <- matrix(0, plot_resolution, plot_resolution)

for (i in 1:plot_resolution) {
  for (j in 1:plot_resolution) {
    z_surf[i, j] <- rosenbrock(x_surf[i], y_surf[j])
  }
}

# Limit z values for better visualization
z_max_vis <- 500
z_surf[z_surf > z_max_vis] <- z_max_vis

# Create the 3D surface plot with perspective
persp_plot <- persp(x_surf, y_surf, z_surf, 
                    theta = 30, phi = 20, # Viewing angles
                    col = "lightblue", shade = 0.5,
                    ticktype = "detailed",
                    xlab = "X", ylab = "Y", zlab = "f(X,Y)",
                    main = "3D Visualization of Gradient Descent on Rosenbrock Function")

# Calculate path z-values
path_z <- sapply(1:length(result$history_x), function(i) {
  min(rosenbrock(result$history_x[i], result$history_y[i]), z_max_vis)
})

# Convert 3D coordinates to 2D perspective for plotting the path
path_points <- trans3d(result$history_x, result$history_y, path_z, pmat = persp_plot)
start_point <- trans3d(result$history_x[1], result$history_y[1], path_z[1], pmat = persp_plot)
end_point <- trans3d(
  result$history_x[length(result$history_x)],
  result$history_y[length(result$history_y)],
  path_z[length(path_z)], 
  pmat = persp_plot
)

# Add the path to the 3D plot
lines(path_points, col = "red", lwd = 2)
points(start_point, col = "green", pch = 19, cex = 1.5)
points(end_point, col = "blue", pch = 19, cex = 1.5)

# Add legend to the main plot area
legend("topright", 
       legend = c("Path", "Start", "End"), 
       col = c("red", "green", "blue"), 
       pch = c(NA, 19, 19), 
       lwd = c(2, NA, NA), 
       cex = 0.8)
