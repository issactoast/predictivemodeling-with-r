rss <- function(s){
    x <- -10:10
    y <- x^2
    
    group1_index <- which(x < s)
    group2_index <- which(x >= s)
    x[group1_index]
    x[group2_index]
    
    pred_group1 <- mean(y[group1_index])
    pred_group2 <- mean(y[group2_index])
    
    result <- sum((y[group1_index] - pred_group1)^2) +
        sum((y[group2_index] - pred_group2)^2)
    result
}
rss(-2)
rss(-9.99)
s[2]
s <- seq(-10, 10, by = 0.01)
rss_result <- sapply(s, rss)

plot(s, rss_result)

s[which.min(rss_result)]
rss(-7.99)

group1_index <- which(x < -7.99)
group2_index <- which(x >= -7.99)

mean(y[group2_index])

plot(x, y)
abline(h = mean(y[group1_index]))
abline(h = mean(y[group2_index]))
abline(v = -7.99)