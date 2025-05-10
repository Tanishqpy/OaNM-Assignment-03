# Generate data
x <- rnorm(100, mean = 1, sd = 2^(1/2))
y <- rnorm(100, mean = 2+3*x, sd = 5^(1/2))

# Loss function as specified in the problem
f_0 <- function(beta, X, y, n) {
  1/(2*n) * sum((y - X %*% beta)^2)
}

# Gradient of the loss function
grad_f <- function(beta, X, y, n) {
  -1/n * t(X) %*% (y - X %*% beta)
}

# Gradient descent function
gradient_descent <- function(X, y, eta=0.01, epsilon=1e-6, max_iter=10000) {
  n <- length(y)
  
  # Initial point beta_0 = [0,0]
  beta <- matrix(c(0, 0), nrow=2)
  
  # Create design matrix X with intercept column
  X_mat <- cbind(1, X)
  
  # Track loss values
  loss_history <- c()
  
  for (i in 1:max_iter) {
    # Current loss
    current_loss <- f_0(beta, X_mat, y, n)
    loss_history <- c(loss_history, current_loss)
    
    # Compute gradient
    gradient <- grad_f(beta, X_mat, y, n)
    
    # Update beta
    beta_new <- beta - eta * gradient
    
    # Check convergence
    if (i > 1) {
      if (abs(loss_history[i-1] - current_loss) < epsilon) {
        cat("Converged after", i, "iterations\n")
        break
      }
    }
    
    beta <- beta_new
  }
  
  if (i == max_iter)
    cat("Maximum iterations reached\n")
  
  # Plot results
  plot(X, y, col="blue", pch=20, main="Gradient Descent for Linear Regression", 
       xlab="X", ylab="Y")
  abline(beta[1], beta[2], col="red", lwd=2)
  
  return(list(
    intercept = beta[1], 
    slope = beta[2], 
    iterations = i,
    loss = current_loss,
    loss_history = loss_history
  ))
}

# Run gradient descent
set.seed(123) # For reproducibility
result <- gradient_descent(x, y)

# Print results
cat("Optimal parameters:\n")
cat("Intercept (β₀):", result$intercept, "\n")
cat("Slope (β₁):", result$slope, "\n")
cat("Final loss value:", result$loss, "\n")
cat("Number of iterations needed:", result$iterations, "\n")

# Compare with lm() function
lm_model <- lm(y ~ x)
cat("\nComparison with lm() function:\n")
cat("lm intercept:", coef(lm_model)[1], "\n")
cat("lm slope:", coef(lm_model)[2], "\n")

# Create convergence plot
par(mfrow=c(1,1))
plot(1:length(result$loss_history), result$loss_history, type="l", col="blue", 
     main="Convergence of Gradient Descent", 
     xlab="Iteration", ylab="Loss Value", 
     lwd=2)
# Add horizontal line at the final loss value
abline(h=result$loss, col="red", lty=2)
text(length(result$loss_history)/2, result$loss*1.1, 
     paste("Final loss:", round(result$loss, 6)), col="red", pos = 3)

# Analytical solution for comparison
X_mat <- cbind(1, x)
beta_analytical <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% y
analytical_loss <- f_0(beta_analytical, X_mat, y, length(y))

cat("\n----- Gradient Descent Summary -----\n")
cat("Final solution:\n")
cat("β₀ =", result$intercept, "\n")
cat("β₁ =", result$slope, "\n")
cat("Iterations required:", result$iterations, "\n")
cat("Final loss value:", result$loss, "\n")
cat("\nComparison with analytical solution:\n")
cat("Analytical β₀ =", beta_analytical[1], "\n")
cat("Analytical β₁ =", beta_analytical[2], "\n")
cat("Analytical loss =", analytical_loss, "\n")

cat("\nObservations:\n")
cat("1. The gradient descent algorithm ", 
    ifelse(result$iterations < 10000, "converged successfully", "reached maximum iterations"),
    " after ", result$iterations, " iterations.\n", sep="")

cat("2. Error between gradient descent and analytical solution: ", 
    round(sqrt(sum((c(result$intercept, result$slope) - beta_analytical)^2)), 6), "\n")

cat("3. Learning rate (eta) was set to:", 0.01, "which is a key factor in convergence speed.\n")

cat("\nPotential difficulties with gradient descent:\n")
cat("- Learning rate selection: Too large may cause divergence, too small leads to slow convergence.\n")
cat("- Sensitivity to initial values and feature scaling.\n")
cat("- May struggle with ill-conditioned problems.\n")
cat("- Local minima issues in non-convex optimization problems (not applicable for linear regression).\n")