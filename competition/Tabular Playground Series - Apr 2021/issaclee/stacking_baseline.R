# library(treesnip)
library(tidymodels)
library(tidyverse)
library(magrittr)
library(skimr)
library(knitr)
theme_set(theme_bw())

# setwd("yourfolderpath")
file_path <- "../input/tabular-playground-series-apr-2021/"
files <- list.files(file_path)
files


train <- read_csv(file.path(file_path, "train.csv")) %>% 
    janitor::clean_names()
test <- read_csv(file.path(file_path, "test.csv")) %>% 
    janitor::clean_names()


dim(train)
dim(test)

index <- 1:ncol(train)
glue::glue("Train set variable {index}: {names(train)}") 

skim(train)
skim(test)

train %>% 
    count(survived) %>% 
    mutate(proportion = n / sum(n)) %>% 
    ggplot(aes(x = factor(survived), y = proportion, 
               fill = factor(survived))) +
    geom_bar(stat = "identity") +
    labs(title = "Bar chart for target variable",
         subtitle = "Variable label: 'survived'",
         x = "Survived") +
    scale_fill_brewer(palette = "Set1",
                      labels = c("No", "Yes")) +
    guides(fill = guide_legend(title = "Survival",
                               ncol = 2, )) +
    theme(legend.position = "bottom")


train$name
train %>% skim()

train$survived <- as.factor(train$survived) 

titanic_recipe <- train %>% 
    recipe(survived ~ .) %>%
    step_mutate(
        last_name = word(name, 1)
    ) %>% 
    step_modeimpute(embarked) %>% 
    step_meanimpute(age, fare) %>% 
    step_rm(passenger_id, cabin, ticket) %>% 
    step_integer(all_nominal(), -all_outcomes()) %>% 
    step_center(all_predictors(), -all_outcomes()) %>% 
    prep(training = train)

train2 <- juice(titanic_recipe)
test2 <- bake(titanic_recipe, new_data = test)




# 5 fold cv
set.seed(2021)
validation_split <- vfold_cv(v = 5, train2,
                             strata = survived)

logitstic_spec <- multinom_reg(
    penalty = tune(),
    mixture = tune()
) %>% 
    set_engine("glmnet") %>% 
    set_mode("classification")

logitstic_spec %>% translate()


param_grid <- grid_latin_hypercube(
    penalty(),
    mixture(),
    size = 3
)
param_grid

logistic_workflow <- workflow() %>%
    add_model(logitstic_spec) %>% 
    add_formula(survived ~ .)

# for stacking
ctrl_res <- control_stack_grid()

library(tictoc)
tic()
logit_tune_result <- logistic_workflow %>% 
    tune_grid(validation_split,
              grid = param_grid,
              metrics = metric_set(accuracy),
              control = ctrl_res)
toc()

# tune_result %>% 
#     collect_metrics()
# 
# 
# tune_best <- tune_result %>% 
#     select_best(metric = "accuracy")
# A tibble: 1 x 3
# penalty mixture .config             
# <dbl>   <dbl> <chr>               
#     1 6.12e-10   0.880 Preprocessor1_Model3

# final_spec <- finalize_model(logitstic_spec, tune_best)
# final_spec
# 
# logistic_workflow %<>% update_model(final_spec)

# random forest
rf_spec <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>% 
    set_engine("ranger",
               num.threads = 10) %>% 
    set_mode("classification")

rf_workflow <- workflow() %>%
    add_model(rf_spec) %>% 
    add_formula(survived ~ .)

rf_param_grid <- grid_random(finalize(mtry(), x = train2[,-1]),
                             min_n(),
                             size = 3)

rf_tune_result <- rf_workflow %>% 
    tune_grid(validation_split,
              grid = rf_param_grid,
              metrics = metric_set(accuracy),
              ctrol = ctrl_res)

titanic_stacking <- 
    stacks() %>% 
    add_candidates(logit_tune_result) %>% 
    add_candidates(rf_tune_result)

# print stacking
titanic_stacking

# as tibble
as_tibble(titanic_stacking) %>% head()

titanic_stacking %<>% 
    blend_predictions() %>% 
    fit_members()

titanic_stacking

# prediction
result <- predict(titanic_stacking, test2)


logistic_fit <- fit(logistic_workflow, data = train2)

result <- predict(logistic_fit, test2, type = "prob")
result

result


submission <- read_csv(file.path(file_path, "sample_submission.csv"),
                       col_types = cols(PassengerId = col_integer(),
                                        Survived = col_integer()))

submission$Survived <- result$.pred_class %>% as.integer() - 1
write.csv(submission, row.names = FALSE,
          "logistic_baseline.csv")

submission %>% head()
