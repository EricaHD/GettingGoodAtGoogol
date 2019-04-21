# DeepGoogol
Final Project for Computational Cognitive Modeling (NYU PSYCH-GA 3405.002 / DS-GS 3001.005)

#### Q Learning Results

| **RESULTS** | **TIME** | | alpha | alpha\_decay | alpha\_step | gamma | epsilon | eps\_decay | s\_cost | q\_learn | q\_key\_fn | q\_key\_params | v\_fn | lo | hi | n\_idx | replace | reward\_fn | reward | n\_games | n\_print | delay | curr\_epoch | curr\_params | lo\_eval | hi\_eval | n\_idx\_eval | replace\_eval | reward\_fn\_eval | reward\_eval | n\_games\_eval | n\_print\_eval | delay\_eval |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 7.4% | 5h,29m | | 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **10000000** | 10000 | 0 | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 19% | 4h,45m | | 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **10000000** | 10000 | 0 | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| Out of memory [1] | 2h,42m | | 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **50000000** | 10000 | 0 | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| Out of memory [2] | 2h,37m | | 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **50000000** | 10000 | 0 | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |

[1] Ran out of 12GB of memory.  Stopped at 26%.  Training victory percentage 3.3%.
[2] Ran out of 12GB of memory.  Stopped at 27%.  Training victory percentage 8.5%.

#### Monte Carlo Results

| **RESULTS** | **TIME** | | gamma | epsilon | eps\_decay | s\_cost | q\_key\_fn | q\_key\_params | v\_fn | lo | hi | n\_idx | replace | reward\_fn | reward | n\_episodes | curr\_epoch | curr\_params | lo\_eval | hi\_eval | n\_idx\_eval | replace\_eval | reward\_fn\_eval | reward\_eval | n\_games\_eval | n\_print\_eval | delay\_eval |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 23% | 1h,20m | | 0.9 | 0.1 | 0.00001 | 0 | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **1000000** | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 27% | 46m | | 0.9 | 0.1 | 0.00001 | 0 | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **1000000** | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 3.2% | 2h,6m | | 0.9 | 0.1 | 0.00001 | 0 | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **100000** | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 3.9% | 2h,39m | | 0.9 | 0.1 | 0.00001 | 0 | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **100000** | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
