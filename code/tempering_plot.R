setwd("~/Desktop/functional_landscapes/code")
library(tidyverse)
library(viridis)

data =read.csv('../data/tot_results.csv') %>% 
  filter()
ggplot(data, 
       aes(x = t, 
           y = ssq))+
  geom_line(aes(group = chain,
                color = log(1+T)))+
  scale_color_viridis(option='turbo')+
  scale_y_continuous(trans='log')
ggsave('../data/tempering_plot_series.pdf', width = 6, height = 3)
