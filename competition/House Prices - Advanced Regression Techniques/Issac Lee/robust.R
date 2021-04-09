#' Robust normalize numeric data
#' 
step_robust_normalize <- function(recipe, 
    ..., 
    role = NA, 
    trained = FALSE, 
    means = NULL,
    sds = NULL,
    na.rm = TRUE,
    skip = FALSE,
    id = rand_id("robust_normalize")) {
    
    terms <- ellipse_check(...) 
    
    add_step(
        recipe, 
        step_robust_normalize_new(
            terms = terms, 
            trained = trained,
            role = role, 
            means = means,
            sds = sds,
            na.rm = na.rm,
            skip = skip,
            id = id
        )
    )
}

step_robust_normalize_new <- 
    function (terms, role, trained, means, sds, na.rm, skip, id){
        step(subclass = "robust_normalize",
             terms = terms,
             role = role,
             trained = trained,
             means = means,
             sds = sds,
             na.rm = na.rm, 
             skip = skip,
             id = id)
    }

prep.step_robust_normalize <- function (x, training, info = NULL, ...) {
    col_names <- terms_select(terms = x$terms, info = info)
    check_type(training[, col_names])
    
    means <- vapply(training[, col_names], median, c(median = 0), 
                    na.rm = x$na.rm)
    sds <- vapply(training[, col_names], IQR, c(IQR = 0), na.rm = x$na.rm)
    sds[sds==0] <- 2
    step_robust_normalize_new(terms = x$terms, role = x$role, trained = TRUE, 
                              means = means, sds = sds/2, na.rm = x$na.rm, skip = x$skip, 
                              id = x$id)
}

bake.step_robust_normalize <- function (object, new_data, ...) {
    res <- sweep(as.matrix(new_data[, names(object$means)]), 
                 2, object$means, "-")
    res <- sweep(res, 2, object$sds, "/")
    res <- tibble::as_tibble(res)
    new_data[, names(object$sds)] <- res
    as_tibble(new_data)
}

print.step_robust_normalize <-
    function(x, width = max(20, options()$width - 30), ...) {
        cat("Centering and scaling for ", sep = "")
        printer(names(x$sds), x$terms, x$trained, width = width)
        invisible(x)
    }

tidy.step_robust_normalize <- function (x, ...) 
{
    if (is_trained(x)) {
        res <- tibble(terms = c(names(x$means), names(x$sds)), 
                      statistic = rep(c("median", "IQR"), each = length(x$sds)), 
                      value = c(x$means, x$sds))
    }
    else {
        term_names <- sel2char(x$terms)
        res <- tibble(terms = term_names, statistic = na_chr, 
                      value = na_dbl)
    }
    res$id <- x$id
    res
}
