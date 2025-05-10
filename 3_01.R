# Quadratic Convex Function Minimization using Gradient Descent

# Define the matrices A and b
A <- matrix(c(2, 0, 0, 4), nrow = 2, byrow = TRUE)
b <- c(-4, -8)

# Objective function f(x) = x^T A x + b^T x
f <- function(x) {
  return(as.numeric(t(x) %*% A %*% x + sum(b * x)))
}

# Gradient of f(x): ∇f(x) = 2Ax + b
gradient_f <- function(x) {
  return(2 * A %*% x + b)
}

# Gradient descent
gradient_descent <- function(x0, eta, epsilon, max_iter = 1000) {
  # Initialize
  x <- x0
  iter <- 0
  grad_norm <- Inf
  
  # Pre-allocate arrays for history
  x_history <- matrix(0, nrow = max_iter + 1, ncol = length(x0))
  f_history <- numeric(max_iter + 1)
  
  # Store initial values
  x_history[1,] <- x0
  f_history[1] <- f(x0)
  
  # Compute initial gradient
  grad <- gradient_f(x)
  grad_norm <- sqrt(sum(grad^2))
  
  while (grad_norm >= epsilon && iter < max_iter) {
    iter <- iter + 1
    
    # Update x
    x <- x - eta * grad
    
    # Store history
    x_history[iter + 1,] <- x
    f_history[iter + 1] <- f(x)
    
    # Update gradient and its norm
    grad <- gradient_f(x)
    grad_norm <- sqrt(sum(grad^2))
    
    # Print periodic updates
    if(iter %% 100 == 0) {
      cat("Iteration:", iter, "Function value:", f_history[iter + 1], "Gradient Norm:", grad_norm, "\n")
    }
  }
  
  # Trim arrays to actual size used
  if(iter < max_iter) {
    x_history <- x_history[1:(iter + 1),, drop = FALSE]  # Keep as matrix with drop = FALSE
    f_history <- f_history[1:(iter + 1)]
  }
  
  return(list(
    x = x,
    f_min = f(x),
    iterations = iter,
    x_history = x_history,
    f_history = f_history,
    converged = (grad_norm < epsilon),
    grad_norm = grad_norm
  ))
}

# Initial point
x0 <- c(1, 1)

# Step size
eta <- 0.1

# Threshold
epsilon <- 1e-6

# Run gradient descent
result <- gradient_descent(x0, eta, epsilon)

# Print results
cat("Minimum found at x =", result$x[1], ",", result$x[2], "\n")
cat("Minimum function value:", result$f_min, "\n")
cat("Number of iterations:", result$iterations, "\n")
cat("Converged:", result$converged, "\n")
cat("Final gradient norm:", result$grad_norm, "\n")

# Plot the loss vs. iterations
# Check if we have more than just the initial point to plot
if(result$iterations > 0) {
  plot(0:result$iterations, result$f_history, type = "l", 
       xlab = "Iterations", ylab = "Function Value", 
       main = "Loss vs. Iterations",
       col = "blue", lwd = 2)
  grid()
} else {
  # If we only have the initial point, plot a single point
  plot(0, result$f_history[1], 
       xlab = "Iterations", ylab = "Function Value", 
       main = "Loss vs. Iterations (Immediate Convergence)",
       col = "blue", pch = 16)
  grid()
  cat("Warning: Gradient descent converged immediately!\n")
  cat("Initial gradient norm may be below epsilon threshold.\n")
}

# Compute analytic minimum for verification
x_min_analytic <- -0.5 * solve(A) %*% b
f_min_analytic <- f(x_min_analytic)
cat("\nAnalytic minimum at x =", x_min_analytic[1], ",", x_min_analytic[2], "\n")
cat("Analytic minimum function value:", f_min_analytic, "\n")

# Create a grid for contour plot 
padding <- 0.5
x1_range <- seq(min(min(result$x_history[,1]), x_min_analytic[1]) - padding, 
               max(max(result$x_history[,1]), x_min_analytic[1]) + padding, 
               length.out = 100)
x2_range <- seq(min(min(result$x_history[,2]), x_min_analytic[2]) - padding, 
               max(max(result$x_history[,2]), x_min_analytic[2]) + padding, 
               length.out = 100)

# Calculate function values on the grid
z_matrix <- matrix(0, nrow = length(x1_range), ncol = length(x2_range))
for (i in 1:length(x1_range)) {
  for (j in 1:length(x2_range)) {
    z_matrix[i, j] <- f(c(x1_range[i], x2_range[j]))
  }
}

# 2D contour plot with descent path
contour(x1_range, x2_range, z_matrix, nlevels = 20, 
        xlab = "x1", ylab = "x2", main = "Gradient Descent Path")
lines(result$x_history[,1], result$x_history[,2], col = "red", type = "o", pch = 16, cex = 0.8)
points(result$x_history[1,1], result$x_history[1,2], col = "blue", pch = 16, cex = 1.5)  # Start
points(result$x_history[nrow(result$x_history),1], 
       result$x_history[nrow(result$x_history),2], col = "green", pch = 16, cex = 1.5)  # End
points(x_min_analytic[1], x_min_analytic[2], col = "purple", pch = 8, cex = 1.5)  # Analytic solution

# Add a legend
legend("topright", legend = c("Path", "Start", "End", "Analytic Min"), 
       col = c("red", "blue", "green", "purple"), 
       pch = c(16, 16, 16, 8), cex = 0.8)

# 3D visualization
persp(x1_range, x2_range, z_matrix, theta = 30, phi = 30, 
      expand = 0.6, col = "lightblue", 
      xlab = "x1", ylab = "x2", zlab = "f(x)", 
      main = "3D Visualization of Quadratic Function")
