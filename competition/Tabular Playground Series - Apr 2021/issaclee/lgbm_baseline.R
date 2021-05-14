## ----setup, include=FALSE------------
knitr::opts_chunk$set(echo = TRUE,
                      message = FALSE,
                      warning = FALSE,
                      fig.align = "center")


## ----load_lib, message=FALSE, warning=FALSE, results='hide'----
# remotes::install_github("curso-r/treesnip")
library(treesnip)
library(tidymodels)
library(tidyverse)
library(magrittr)
library(skimr)
library(knitr)
theme_set(theme_bw())


## ------------------------------------
file_path <- "../input/tabular-playground-series-apr-2021/"
files <- list.files(file_path)
files


## ---- message=FALSE------------------
train <- read_csv(file.path(file_path, "train.csv")) %>% 
  janitor::clean_names()
test <- read_csv(file.path(file_path, "test.csv")) %>% 
  janitor::clean_names()


## ------------------------------------
dim(train)
dim(test)


## ----class.source = 'fold-hide'------
index <- 1:ncol(train)
glue::glue("Train set variable {index}: {names(train)}") 


## ----class.source = 'fold-hide'------
index <- 1:ncol(test)
glue::glue("Train set variable {index}: {names(test)}") 


## ------------------------------------
head(train) %>% kable()
head(train) %>% kable()


## ------------------------------------
skim(train)


## ------------------------------------
skim(test)


## ----message=FALSE, class.source = 'fold-hide'----
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


## ------------------------------------
all_data <- bind_rows(train, test)
all_data %>% head()
names(all_data)
dim(all_data)


## ------------------------------------
titanic_recipe <- all_data %>% 
    recipe(survived ~ .) %>%
    step_mutate(
        survived = factor(survived),
        last_name = word(name, 1)
        ) %>% 
    step_modeimpute(embarked) %>% 
    step_meanimpute(age, fare) %>% 
    step_rm(passenger_id, cabin, ticket) %>% 
    step_integer(all_nominal(), -all_outcomes()) %>% 
    step_center(all_predictors(), -all_outcomes()) %>% 
    prep(training = all_data)

all_data2 <- juice(titanic_recipe)
all_data2 %>% dim()
all_data2 %>% head()


## ------------------------------------
train_index <- seq_len(nrow(train))
train2 <- all_data2[train_index,]
test2 <- all_data2[-train_index,]


## ------------------------------------
set.seed(2021)

# 10 fold cv
validation_split <- vfold_cv(v = 5, train2, strata = survived)


## ------------------------------------
# Main Arguments:
#   mtry = 1
#   trees = 10000
#   min_n = 38
#   tree_depth = 10
#   learn_rate = 0.005
#   loss_reduction = 0.0672681981394635
#   sample_size = 0.45471338326586
#   stop_iter = 10

lightgbm_spec <- boost_tree(
    trees = 10000, 
    tree_depth = 100, 
    mtry = tune(),
    min_n = tune(), 
    loss_reduction = 0.0672681981394635,  
    sample_size = 0.45471338326586, 
    learn_rate = 0.005,
    stop_iter = 10,
) %>% 
    set_engine('lightgbm',
               num_leaves = 60,
               # categorical_feature = c(1, 2, 5, 6, 8, 10),
               num_threads = 10) %>% 
    set_mode('classification')

lightgbm_spec %>% translate()

param_grid <- grid_random(
    finalize(mtry(), train2[-1]),
    min_n(), 
    # loss_reduction(),
    # sample_size = sample_prop(range = c(0.4, 1)),
    size = 15
) %>% filter(mtry > 3)
param_grid


## ------------------------------------
lightgbm_workflow <- workflow() %>%
    add_model(lightgbm_spec) %>% 
    add_formula(survived ~ .)


## ------------------------------------
library(tictoc)
tic()
tune_result <- lightgbm_workflow %>% 
  tune_grid(validation_split,
            grid = param_grid,
            metrics = metric_set(accuracy))
toc()


## ------------------------------------
tune_result$.notes[[1]]$.notes
tune_result %>% 
    collect_metrics()


## ------------------------------------
tune_result %>% show_best()


## ------------------------------------
tune_best <- tune_result %>% select_best(metric = "accuracy")
final_spec <- finalize_model(lightgbm_spec, tune_best)
final_spec
# Boosted Tree Model Specification (classification)
# 
# Main Arguments:
#   mtry = 1
#   trees = 10000
#   min_n = 38
#   tree_depth = 10
#   learn_rate = 0.005
#   loss_reduction = 0.0672681981394635
#   sample_size = 0.45471338326586
#   stop_iter = 10
# 
# Engine-Specific Arguments:
#   num_leaves = 60
#   num_threads = 10
# 
# Computational engine: lightgbm 


## ------------------------------------
lightgbm_workflow %<>% update_model(final_spec)
lightgbm_fit <- fit(lightgbm_workflow, data = train2)


## ------------------------------------
result <- predict(lightgbm_fit, test2)
result %>% head()


## ------------------------------------
submission <- read_csv(file.path(file_path, "sample_submission.csv"),
                       col_types = cols(PassengerId = col_integer(),
                                        Survived = col_integer()))
submission$Survived <- result$.pred_class %>% as.integer() - 1
write.csv(submission, row.names = FALSE,
          "lightgbm_baseline.csv")
submission %>% head()

# write.csv(result2, row.names = FALSE,
#           "group_mean.csv")

