x <- rnorm(10, mean = 10, sd = 3)
x

mean(x)

1000
sample(x, length(x), replace = TRUE)
mean()

result <- rep(0, 1000)
x <- rnorm(10, mean = 10, sd = 3)

for (i in 1:1000) {
    result[i] <- mean(sample(x, length(x), replace = TRUE))
}
sd(result)

# bagging
sample(mtcars, length(mtcars), replace = TRUE)
mtcars %>% dim()
sample()

my_f <- function() {
    index <- sample(1:nrow(mtcars), nrow(mtcars), replace = TRUE)
    mtcars[index, ]    
}



