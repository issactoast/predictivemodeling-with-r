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
