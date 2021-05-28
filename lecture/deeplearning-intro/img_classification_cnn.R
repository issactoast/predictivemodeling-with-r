library(torch)
library(torchvision)
library(torchdatasets)

library(pins)
library(dplyr)
library(ggplot2)

board_register_kaggle(token = "./kaggle.json")
device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
# device <- "cpu"

train_transforms <- function(img) {
    img %>%
        # first convert image to tensor
        transform_to_tensor() %>%
        # then move to the GPU (if available)
        (function(x) x$to(device = device)) %>%
        # data augmentation
        transform_random_resized_crop(size = c(224, 224)) %>%
        # data augmentation
        transform_color_jitter() %>%
        # data augmentation
        transform_random_horizontal_flip() %>%
        # normalize according to what is expected by resnet
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}


valid_transforms <- function(img) {
    img %>%
        transform_to_tensor() %>%
        (function(x) x$to(device = device)) %>%
        transform_resize(256) %>%
        transform_center_crop(224) %>%
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

test_transforms <- valid_transforms

train_ds <- bird_species_dataset("data", download = FALSE, transform = train_transforms)
valid_ds <- bird_species_dataset("data", split = "valid", transform = valid_transforms)
test_ds <- bird_species_dataset("data", split = "test", transform = test_transforms)

train_ds$.length()
valid_ds$.length()
test_ds$.length()

batch_size <- 64

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = batch_size)
test_dl <- dataloader(test_ds, batch_size = batch_size)

batch <- train_dl$.iter()$.next()

batch$x$size()

# 그림 그리기 위해서
class_names <- test_ds$classes
classes <- batch[[2]]

library(dplyr)

images <- as_array(batch[[1]]$to(device = "cpu")) %>% 
    aperm(perm = c(1, 3, 4, 2))
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)
images <- std * images + mean
images <- images * 255
images[images > 255] <- 255
images[images < 0] <- 0

par(mfcol = c(4,6), mar = rep(1, 4))

images %>%
    purrr::array_tree(1) %>%
    purrr::set_names(class_names[as_array(classes)]) %>%
    purrr::map(as.raster, max = 255) %>%
    purrr::iwalk(~{plot(.x); title(.y)})


model <- model_resnet18(pretrained = TRUE)

model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

# 마지막 단 조정하기
num_features <- model$fc$in_features
model$fc <- nn_linear(in_features = num_features, out_features = length(class_names))


b <- train_dl$.iter()$.next()

model <- model$to(device = device)

model(b[[1]])

criterion <- nn_cross_entropy_loss()

optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9)

num_epochs <- 3

scheduler <- optimizer %>% 
    lr_one_cycle(max_lr = 0.05, epochs = num_epochs, 
                 steps_per_epoch = train_dl$.length())

train_batch <- function(b) {
    
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- criterion(output, b[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    scheduler$step()
    loss$item()
    
}

valid_batch <- function(b) {
    
    output <- model(b[[1]])
    loss <- criterion(output, b[[2]]$to(device = device))
    loss$item()
}

for (epoch in 1:num_epochs) {
    
    model$train()
    train_losses <- c()
    
    # b <- train_dl$.iter()$.next()
    coro::loop(for (b in train_dl) {
        loss <- train_batch(b)
        train_losses <- c(train_losses, loss)
    })
    
    model$eval()
    valid_losses <- c()
    
    coro::loop(for (b in valid_dl) {
        loss <- valid_batch(b)
        valid_losses <- c(valid_losses, loss)
    })
    
    cat(sprintf("\nLoss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}
