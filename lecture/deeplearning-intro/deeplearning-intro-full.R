library(palmerpenguins)
penguins %>% head(5)


library(recipes)

penguin_data <- penguins %>% 
    recipe(species ~ .) %>%
    step_impute_mode(all_nominal()) %>% 
    step_impute_mean(all_numeric()) %>% 
    step_dummy(all_nominal(), -all_outcomes()) %>% 
    step_integer(all_nominal()) %>% 
    step_normalize(all_predictors(), -all_outcomes()) %>% 
    prep() %>% juice()

penguin_data %>% dim()

penguin_data %>% head()

library(rsample)
split_data <- initial_split(penguin_data, prop = 0.8) 
penguin_data_train <- training(split_data)
penguin_data_test <- testing(split_data)

penguin_dataset <- dataset(
    name = "penguin_data",
    initialize = function() {
        self$data <- torch_tensor(as.matrix(select(penguin_data, species, everything())))
    },
    .getitem = function(index) {
        x <- self$data[index, 2:9]
        y <- self$data[index, 1]
        
        list(x, y)
    },
    .length = function() {
        self$data$size()[[1]]
    }
)

torch_penguin_data <- penguin_dataset()
torch_penguin_data

torch_penguin_data$.getitem(1:6)
penguin_dl <- dataloader(torch_penguin_data, batch_size = 8)
penguin_dl$.length()

b <- penguin_dl$.iter()$.next()
length(b)
b[[1]]

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
my_net <- my_net$to(device = device)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(my_net$parameters, 
                       lr = 0.1, momentum = 0.9)
num_epochs <- 5

train_batch <- function(b) {
    
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- criterion(output, b[[2]]$to(device = device,
                                        dtype  = torch_long()))
    loss$backward()
    optimizer$step()
    scheduler$step()
    loss$item()
    
}

valid_batch <- function(b) {
    
    output <- model(b[[1]])
    loss <- criterion(output, b[[2]]$to(device = device, 
                                        dtype  = torch_long()))
    loss$item()
}

for (epoch in 1:num_epochs) {
    my_net$train()
    train_losses <- c()  
    
    coro::loop(for (b in penguin_dl) {
        loss <- train_batch(b)
        train_losses <- c(train_losses, loss)
    })
    
    model$eval()
    valid_losses <- c()
    
    coro::loop(for (b in valid_dl) {
        loss <- valid_batch(b)
        valid_losses <- c(valid_losses, loss)
    })
    
    if (epoch %% 100 == 0){
        cat(sprintf("\nLoss at epoch %d: training: %3f, validation: %3f\n",
                    epoch, mean(train_losses), mean(valid_losses)))
    }
    
}