library(treesnip)
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




