library(rcdk)
library(dplyr)
library(rcdklibs)
library(Rtsne)
library(stats)
library(ggplot2)
library(proxy)
library(plotly)



smiles_to_fingerprint <- function(smiles) {
  tryCatch({
    mol <- parse.smiles(smiles)[[1]]
    fp <- get.fingerprint(mol, type='extended')
    return(fingerprint_to_vector(fp))
  }, error = function(e) {
    return(rep(NA, length_of_fingerprint))
  })
}

fingerprint_to_vector <- function(fp) {
  vec <- rep(0, length_of_fingerprint)
  if (!is.null(fp)) {
    on_bits <- fp@bits
    vec[on_bits] <- 1
  }
  return(vec)
}

calculate_cluster_similarity_matrix <- function(fingerprints, cluster_membership, cluster_id) {
 
  in_cluster_indices <- which(cluster_membership == cluster_id)

  
  cluster_fingerprints <- fingerprints[in_cluster_indices, ]

 
  similarity_matrix <- proxy::simil(cluster_fingerprints, method = "Tanimoto", by_rows = TRUE)
  return(similarity_matrix)
}


find_representative_molecule <- function(cluster_id, cluster_membership, fingerprints, smileses) {

  similarity_matrix <- calculate_cluster_similarity_matrix(fingerprints, cluster_membership, cluster_id)

 
  average_similarity <- apply(similarity_matrix, 1, mean)


  representative_index <- which.max(average_similarity)
  representative_smiles <- smileses[cluster_membership == cluster_id][representative_index]
  return(representative_smiles)
}



options(java.parameters = "-Xmx8g")
.jinit(parameters = "-Xmx8g")


base_dir <- normalizePath(dirname(rstudioapi::getSourceEditorContext()$path), "/", mustWork = TRUE)
filename <- "1.csv"
file_path <- file.path(base_dir, filename)


length_of_fingerprint <- 2048


data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE) %>%
  distinct(Smiles, .keep_all = TRUE)


fingerprints <- t(sapply(data$Smiles, smiles_to_fingerprint))

if (all(is.na(fingerprints))) {
  stop("No fingerprints, please check SMILES")
}

set.seed(42)
tsne_results <- Rtsne(fingerprints, dims = 3, perplexity=30, theta=0.0, check_duplicates=FALSE)
tsne_coords <- as.data.frame(tsne_results$Y)
colnames(tsne_coords) <- c("V1", "V2", "V3")

dist_matrix <- dist(tsne_coords)
hc <- hclust(dist_matrix, method = "ward.D2")
clusters <- cutree(hc, k = 3) # Adjust 'k' as needed
tsne_coords$cluster <- as.factor(clusters)


representatives <- sapply(unique(clusters), function(c) {
  find_representative_molecule(c, clusters, fingerprints, data$Smiles)
})

representatives <- unique(representatives)


clustered_data <- cbind(data, tsne_coords, cluster = clusters)

clustered_data$is_representative <- FALSE
for (rep_smiles in representatives) {
  clustered_data$is_representative[clustered_data$Smiles == rep_smiles] <- TRUE
}


if(any(table(clustered_data$cluster[clustered_data$is_representative]) > 1)) {
  stop("There are clusters with more than one representative molecule marked.")
}


plot <- plot_ly(data = clustered_data, x = ~V1, y = ~V2, z = ~V3, color = ~cluster,
                colors = RColorBrewer::brewer.pal(8, "Set2"), type = "scatter3d", mode = "markers+text",
                marker = list(size = 2.2, 
                              opacity = 0.35,
                              line = list(color = '#000000', width = 0.4)),
                text = ~ifelse(clustered_data$is_representative, "REP", ""),
                textfont = list(color = ifelse(clustered_data$is_representative, "NA", "black"), 
                                size = ifelse(clustered_data$is_representative, NA, 16)), 
                hoverinfo = 'text') %>% 
  layout(title = "3D t-SNE Clustering",
         scene = list(xaxis = list(title = "t-SNE Dimension 1"),
                      yaxis = list(title = "t-SNE Dimension 2"),
                      zaxis = list(title = "t-SNE Dimension 3")),
         scene = list(camera = list(eye = list(x = 2, y = 2, z = 0.5)))) 


plot

write.csv(clustered_data, file.path(base_dir, "clustered_data.csv"), row.names = FALSE)

