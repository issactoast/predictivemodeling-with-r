## ----load_lib, message=FALSE, warning=FALSE, results='hide'----
library(tidymodels)
library(tidyverse)
library(magrittr)
library(skimr)
theme_set(theme_bw())


## ----------------------------------------------------
file_path <- "../input/walmart-recruiting-store-sales-forecasting/"
files <- list.files(file_path)
files


## ---- message=FALSE----------------------------------
train <- read_csv(file.path(file_path, "train.csv")) %>% 
  janitor::clean_names()
test <- read_csv(file.path(file_path, "test.csv")) %>% 
  janitor::clean_names()
features <- read_csv(file.path(file_path, "features.csv")) %>% 
  janitor::clean_names()
stores <- read_csv(file.path(file_path, "stores.csv")) %>% 
  janitor::clean_names()


## ----------------------------------------------------
dim(train)
dim(test)


## ----------------------------------------------------
names(train)
names(test)

## ----message=FALSE, class.source = 'fold-hide'-------
train %>% 
    ggplot(aes(x = sign(weekly_sales) * (abs(weekly_sales))^(1/5))) +
    geom_histogram() +
    labs(title = "Transformed distribution of weekly sales 2",
         x = "weekly_sales")


## ----------------------------------------------------
all_data <- bind_rows(train, test)
all_data %>% head()

## ----------------------------------------------------
walmart_recipe <- all_data %>% 
    recipe(weekly_sales ~ .) %>%
    step_mutate(year = lubridate::year(date)) %>%   
    step_mutate(month = lubridate::month(date)) %>%
    step_mutate(week = lubridate::week(date)) %>% 
    step_rm(date) %>%
    prep(training = all_data)

print(walmart_recipe)


## ----------------------------------------------------
all_data2 <- juice(walmart_recipe)
all_data2 %>% dim()
all_data2 %>% head()


## ----------------------------------------------------
train_index <- seq_len(nrow(train))
train2 <- all_data2[train_index,]
test2 <- all_data2[-train_index,]

# train2_isholiday <- train2 %>% filter(is_holiday == TRUE)

# 튜닝을 위한 validation 데이터 설정
set.seed(2021)

validation_split <- vfold_cv(train2, v = 3)
# validation_split <- validation_split(train2, prop = 0.3, 
#                                      strata = weekly_sales)

## ----------------------------------------------------
cores <- parallel::detectCores() -1

tune_spec <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 1000) %>% 
    set_engine("ranger", 
               num.threads = cores,
               verbose = TRUE) %>% 
    set_mode("regression")

param_grid <- grid_random(finalize(mtry(), x = train2[,-1]),
                          min_n(),
                          size = 10)

# param_grid <- grid_regular(finalize(mtry(), x = train2[,-1]),
#                           min_n(),
#                           levels = list(mtry = 10, min_n = 5))
# param_grid$mtry %>% unique()
param_grid %<>% filter(mtry >= 3)

## ----------------------------------------------------
workflow <- workflow() %>%
  add_model(tune_spec) %>% 
  add_formula(weekly_sales ~ .)

workflow

## ----tunerf------------------------------------------
library(tictoc)
tic()
tune_result <- workflow %>% 
  tune_grid(validation_split,
            grid = param_grid,
            metrics = metric_set(mae, rmse))
toc()


## ----------------------------------------------------
# tune_result$.notes
tune_result %>% 
  collect_metrics()


## ----message=FALSE-----------------------------------
tune_result %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>% 
  ggplot(aes(mtry, mean, color = .metric)) +
  geom_line(size = 1.5) +
  scale_x_log10() +
  theme(legend.position = "none") +
  labs(title = "MAE")


## ----------------------------------------------------
tune_result %>% show_best(metric = "mae")


## ----------------------------------------------------
tune_best <- tune_result %>% select_best(metric = "mae")
tune_best$mtry
tune_best$min_n

final_spec <- finalize_model(tune_spec, tune_best)
final_spec

## ----trainrf, message=FALSE, warning=FALSE-----------
workflow %<>% update_model(final_spec)
rf_fit <- fit(workflow, data = train2)

## ----warning=FALSE-----------------------------------
# test2$group_mean[is.na(test2$group_mean)] <- 0
result <- predict(rf_fit, test2)
result %>% head()
result

## ----------------------------------------------------
submission <- read_csv(file.path(file_path, "sampleSubmission.csv"))
submission$Weekly_Sales <- result$.pred
write.csv(submission, row.names = FALSE,
          "rf_baseline_april15.csv")
submission %>% head()

