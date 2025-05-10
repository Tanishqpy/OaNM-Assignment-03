# Implementation of gradient descent for Normal Distribution MLE

# Load necessary libraries
library(ggplot2)
library(reshape2)

# Function to compute negative log-likelihood
neg_log_likelihood <- function(mu, sigma, data) {
  n <- length(data)
  sum_term <- sum(log(sigma) + (data - mu)^2 / (2 * sigma^2))
  return(sum_term)
}

# Function to compute gradients
compute_gradient <- function(mu, sigma, data) {
  d_mu <- -sum((data - mu) / sigma^2)
  d_sigma <- sum(1/sigma - (data - mu)^2 / sigma^3)
  return(c(d_mu, d_sigma))
}

# Gradient descent implementation
gradient_descent <- function(data, mu_init = 0, sigma_init = 1, eta = 0.01, 
                             epsilon = 1e-5, max_iter = 10000) {
  # Initialize parameters
  mu <- mu_init
  sigma <- sigma_init
  
  # Track history for plotting
  history <- data.frame(
    iteration = 0,
    mu = mu,
    sigma = sigma,
    loss = neg_log_likelihood(mu, sigma, data),
    grad_norm = NA
  )
  
  for (iter in 1:max_iter) {
    # Compute gradient
    grad <- compute_gradient(mu, sigma, data)
    grad_norm <- sqrt(sum(grad^2))
    
    # Check convergence
    if (grad_norm < epsilon) {
      cat("Converged after", iter, "iterations.\n")
      break
    }
    
    # Update parameters
    mu <- mu - eta * grad[1]
    sigma <- sigma - eta * grad[2]
    
    # Ensure sigma remains positive
    sigma <- max(sigma, 1e-6)
    
    # Store current state
    history <- rbind(history, data.frame(
      iteration = iter,
      mu = mu,
      sigma = sigma,
      loss = neg_log_likelihood(mu, sigma, data),
      grad_norm = grad_norm
    ))
  }
  
  if (iter == max_iter) {
    cat("Maximum iterations reached without convergence.\n")
  }
  
  return(list(
    mu = mu,
    sigma = sigma,
    history = history,
    iterations = iter
  ))
}

# Function to visualize results
plot_results <- function(result, data) {
  # Plot loss vs iterations
  p1 <- ggplot(result$history, aes(x = iteration, y = loss)) +
    geom_line() +
    labs(title = "Loss vs Iterations",
         x = "Iteration",
         y = "Negative Log-Likelihood") +
    theme_minimal()
  
  # Plot descent path in parameter space
  p2 <- ggplot(result$history, aes(x = mu, y = sigma)) +
    geom_path() +
    geom_point(size = 3, color = "red", data = result$history[1,]) +
    geom_point(size = 3, color = "blue", 
               data = result$history[nrow(result$history),]) +
    labs(title = "Gradient Descent Path",
         x = "μ", y = "σ") +
    theme_minimal()
  
  # 3D visualization of the loss function
  if (requireNamespace("plotly", quietly = TRUE)) {
    mu_range <- seq(min(result$history$mu) - 0.5, max(result$history$mu) + 0.5, length.out = 30)
    sigma_range <- seq(min(result$history$sigma) - 0.5, max(result$history$sigma) + 0.5, length.out = 30)
    
    # Create grid for 3D plot
    grid <- expand.grid(mu = mu_range, sigma = sigma_range)
    grid$loss <- apply(grid, 1, function(row) {
      neg_log_likelihood(row["mu"], row["sigma"], data)
    })
    
    # Create 3D surface plot
    p3 <- plotly::plot_ly(
      x = mu_range, 
      y = sigma_range, 
      z = matrix(grid$loss, nrow = length(mu_range), byrow = FALSE),
      type = "surface"
    ) %>%
      plotly::add_trace(
        x = result$history$mu,
        y = result$history$sigma,
        z = result$history$loss,
        type = "scatter3d",
        mode = "lines+markers",
        line = list(color = "red", width = 6),
        marker = list(size = 4, color = "red")
      ) %>%
      plotly::layout(
        title = "Loss Surface and Descent Path",
        scene = list(
          xaxis = list(title = "μ"),
          yaxis = list(title = "σ"),
          zaxis = list(title = "Loss")
        )
      )
    
    print(p3)
  }
  
  print(p1)
  print(p2)
}

# Main execution
main <- function() {
  # Read data from the provided CSV file
  file_path <- "/home/avstr/Tanishq Laptop/Tanishq/BSDS/Semester 2/OaNM/synthetic data - problem-4.csv"
  data <- read.csv(file_path)
  data <- data$x  # Extract the "x" column from the CSV
  
  # Set parameters for gradient descent
  mu_init <- 0      # Initial μ
  sigma_init <- 1   # Initial σ
  eta <- 0.01       # Step size
  epsilon <- 1e-5   # Convergence threshold
  
  # Run gradient descent
  result <- gradient_descent(data, mu_init, sigma_init, eta, epsilon)
  
  # Print results
  cat("\nFinal parameters:\n")
  cat("μ =", result$mu, "\n")
  cat("σ =", result$sigma, "\n")
  cat("\nAnalytical solution (for comparison):\n")
  cat("μ =", mean(data), "\n")
  cat("σ =", sd(data) * sqrt((length(data)-1)/length(data)), "\n")
  
  # Plot results
  plot_results(result, data)
}

# Run the main function
main()
