# beta  = (X^T X)^{-1} X^T y

mtcars
head(mtcars)

y <- as.vector(mtcars[,1])
y

X <- as.matrix(mtcars[,2:4])
X <- cbind(1, X)

# beta?
solve(t(X) %*% X) %*% t(X) %*% y

result <- lm(mpg ~ cyl + disp + hp, data = mtcars)
result$coefficients
