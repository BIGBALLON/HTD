th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape HTLR -optMemory 3 | tee log_HTLR_1.txt
# th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape cosine -optMemory 3 | tee log_cos_1.txt
# th main.lua -netType densenet -dataset cifar100 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape HTLR -optMemory 3 | tee log_HTLR_1.txt
# th main.lua -netType densenet -dataset cifar100 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape cosine -optMemory 3 | tee log_cos_1.txt

# th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape HTLR -optMemory 3 | tee log_HTLR_2.txt
# th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape cosine -optMemory 3 | tee log_cos_2.txt
# th main.lua -netType densenet -dataset cifar100 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape HTLR -optMemory 3 | tee log_HTLR_2.txt
# th main.lua -netType densenet -dataset cifar100 -batchSize 64 -nEpochs 300 -depth 250 -growthRate 24 -nGPU 2 -lrShape cosine -optMemory 3 | tee log_cos_2.txt
