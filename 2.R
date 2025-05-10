# Load the data
data <- read.csv("/home/avstr/Tanishq Laptop/Tanishq/BSDS/Semester 2/OaNM/synthetic data - problem-2.csv")

# Extract features and target
X <- as.matrix(data[, c("x1", "x2")])
y <- data$y

# Sigmoid function
sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

# Negative log-likelihood function
neg_log_likelihood <- function(beta) {
  sum_nll <- 0
  for (i in 1:nrow(X)) {
    z <- y[i] * (X[i,] %*% beta)
    sum_nll <- sum_nll + log(1 + exp(-z))
  }
  return(sum_nll)
}

# Gradient of the negative log-likelihood
gradient <- function(beta) {
  grad <- rep(0, length(beta))
  for (i in 1:nrow(X)) {
    z <- y[i] * (X[i,] %*% beta)
    grad <- grad - y[i] * X[i,] * (1 - sigmoid(z))
  }
  return(grad)
}

# Gradient descent
gradient_descent <- function(initial_beta, eta, epsilon, max_iter = 10000) {
  beta <- initial_beta
  iter <- 0
  grad_norm <- Inf
  
  # Store values for visualization
  beta_history <- matrix(0, nrow=max_iter, ncol=length(beta))
  loss_history <- numeric(max_iter)
  
  while(grad_norm > epsilon && iter < max_iter) {
    iter <- iter + 1
    
    grad <- gradient(beta)
    beta <- beta - eta * grad
    
    grad_norm <- sqrt(sum(grad^2))
    
    beta_history[iter,] <- beta
    loss_history[iter] <- neg_log_likelihood(beta)
    
    if(iter %% 500 == 0) {
      cat("Iteration:", iter, "Loss:", loss_history[iter], "Gradient Norm:", grad_norm, "\n")
    }
  }
  
  if(iter < max_iter) {
    beta_history <- beta_history[1:iter,]
    loss_history <- loss_history[1:iter]
  }
  
  return(list(beta=beta, iter=iter, beta_history=beta_history, loss_history=loss_history))
}

# Initial point, step size, and threshold as given in the problem
beta_init <- c(0, 0)  # Initial point β₀ = [0, 0]
eta <- 0.05          # Step size η = 0.05
epsilon <- 1e-5      # Threshold ε = 10⁻⁵

# Run gradient descent
cat("Running gradient descent for logistic regression...\n")
result <- gradient_descent(beta_init, eta, epsilon)

# Print results
cat("\nFinal beta values:", result$beta, "\n")
cat("Number of iterations required:", result$iter, "\n")
cat("Final negative log-likelihood:", neg_log_likelihood(result$beta), "\n")

# Calculate accuracy
predictions <- sigmoid(X %*% result$beta) > 0.5
accuracy <- mean((predictions == 1) == (y == 1))
cat("Accuracy:", accuracy * 100, "%\n")

# Visualization
library(ggplot2)

# Create a plot showing convergence of the loss function
df_convergence <- data.frame(
  Iteration = 1:length(result$loss_history),
  Loss = result$loss_history
)
p1 <- ggplot(df_convergence, aes(x=Iteration, y=Loss)) +
  geom_line() +
  theme_minimal() +
  ggtitle("Convergence of Negative Log-Likelihood") +
  xlab("Iteration") +
  ylab("Loss")
print(p1)

# Create decision boundary plot
# Generate grid for visualization
x1_range <- seq(min(X[,1]) - 0.5, max(X[,1]) + 0.5, length.out=100)
x2_range <- seq(min(X[,2]) - 0.5, max(X[,2]) + 0.5, length.out=100)
grid <- expand.grid(x1=x1_range, x2=x2_range)
grid_matrix <- as.matrix(grid)

# Calculate probabilities for the grid
probs <- sigmoid(grid_matrix %*% result$beta)
grid$probability <- probs

# Plot decision boundary and data points
p2 <- ggplot() +
  geom_raster(data=grid, aes(x=x1, y=x2, fill=probability), alpha=0.3) +
  geom_contour(data=grid, aes(x=x1, y=x2, z=probability), breaks=0.5, color="black", linewidth=1) +
  geom_point(data=data.frame(X, y=as.factor(y)), aes(x=x1, y=x2, color=y), size=3) +
  scale_fill_gradient(low="blue", high="red") +
  scale_color_manual(values=c("blue", "red")) +
  theme_minimal() +
  ggtitle("Logistic Regression Decision Boundary") +
  xlab("x1") +
  ylab("x2")
print(p2)
