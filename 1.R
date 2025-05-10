# Generate data
x <- rnorm(100, mean = 1, sd = 2^(1/2))
y <- rnorm(100, mean = 2+3*x, sd = 5^(1/2))
epsilon <- 1e-6

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
  
  # Track beta values (for visualization of descent path)
  beta_history <- matrix(0, nrow=max_iter, ncol=2)
  
  # Track gradient norms
  grad_norm_history <- c()
  
  for (i in 1:max_iter) {
    # Current loss
    current_loss <- f_0(beta, X_mat, y, n)
    loss_history <- c(loss_history, current_loss)
    
    # Store current beta
    beta_history[i,] <- t(beta)
    
    # Compute gradient
    gradient <- grad_f(beta, X_mat, y, n)
    
    # Compute gradient norm
    grad_norm <- sqrt(sum(gradient^2))
    grad_norm_history <- c(grad_norm_history, grad_norm)
    
    # Check convergence based on gradient norm
    if (grad_norm < epsilon) {
      cat("Converged after", i, "iterations (gradient norm < epsilon)\n")
      break
    }
    
    # Update beta
    beta_new <- beta - eta * gradient
    
    # Alternative convergence check based on loss difference
    if (i > 1) {
      if (abs(loss_history[i-1] - current_loss) < epsilon) {
        cat("Converged after", i, "iterations (loss difference < epsilon)\n")
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
    loss_history = loss_history,
    beta_history = beta_history[1:i,],
    grad_norm_history = grad_norm_history
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

# Create a plot of the descent path
par(mfrow=c(1,2))

# Plot 1: Parameter path
plot(result$beta_history[,1], result$beta_history[,2], type="b", col="blue",
     xlab="Intercept (β₀)", ylab="Slope (β₁)", main="Parameter Descent Path",
     pch=20, cex=0.6)
points(result$beta_history[1,1], result$beta_history[1,2], col="green", pch=19, cex=1.5)  # Start
points(result$intercept, result$slope, col="red", pch=19, cex=1.5)  # End
points(beta_analytical[1], beta_analytical[2], col="purple", pch=19, cex=1.5)  # Analytical
legend("topright", legend=c("Path", "Start", "End", "Analytical"), 
       col=c("blue", "green", "red", "purple"), pch=c(20,19,19,19), cex=0.8)

# Plot 2: Gradient norm over iterations
plot(1:length(result$grad_norm_history), result$grad_norm_history, type="l", 
     col="darkgreen", xlab="Iteration", ylab="Gradient Norm", 
     main="Gradient Norm vs Iterations", lwd=2)
abline(h=epsilon, col="red", lty=2)
text(length(result$grad_norm_history)/2, epsilon*1.5, 
     paste("Epsilon threshold:", epsilon), col="red", pos=3)

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