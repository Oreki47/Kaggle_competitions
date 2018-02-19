library(igraph) 
library(hydroGOF)

#
# Problem 2 
#
df <- read.table("actor_network.csv", sep =",")
m  <- as.matrix(df)
rm(df)
g <- graph.data.frame(m[, 1:2], directed=TRUE)
E(g)$weight <- m[, 3]
rm(m)
gc()

# checks
is.connected(g)
is.weighted(g)
is.directed(g)

# page_rank
pg <- page.rank(g, algo = "prpack",vids = V(g), 
                directed = TRUE, damping = 0.85)

# Max 10
max_10 <- order(pg$vector, decreasing = TRUE)[0:10]
max_10_val <- pg$vector[max_10]
write.table(max_10_val,file = "max_10.txt", sep = ",")
famous_10 <- c('29740', '13719', '46856', '74894', '42814',
               '113030', '30534', '35587', '17009', '30930')
for (i in famous_10)
{
  print(pg$vector[which(names(pg$vector) == i)])
}
