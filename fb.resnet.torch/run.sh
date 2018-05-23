echo test > start.txt
th main.lua -depth 18 -batchSize 256 -nGPU 4 -nThreads 24 -shareGradInput true -data ~/torch/dataset | tee log.txt  
