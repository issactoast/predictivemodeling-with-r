devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.25.1/catboost-R-Windows-0.25.1.tgz', 
                      INSTALL_opts = c("--no-multiarch"))

library(catboost)

dataset = matrix(c(1900,7,
                   1896,1,
                   1896,41),
                 nrow=3, 
                 ncol=2, 
                 byrow = TRUE)
label_values = c(0,1,1)

fit_params <- list(iterations = 100,
                   loss_function = 'Logloss',
                   task_type = 'GPU')

pool = catboost.load_pool(dataset, label = label_values)

tictoc::tic()
model <- catboost.train(pool, params = fit_params)
tictoc::toc()


library(tidymodels)
library(treesnip)

cat_model <- boost_tree(mode="regression") %>% 
    set_engine("catboost", task_type = "GPU") 

translate(cat_model)
