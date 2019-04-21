# DeepGoogol
Final Project for Computational Cognitive Modeling (NYU PSYCH-GA 3405.002 / DS-GS 3001.005)

#### Q Learning Results

| alpha | alpha\_decay | alpha\_step | gamma | epsilon | eps\_decay | s\_cost | q\_learn | q\_key\_fn | q\_key\_params | v\_fn | lo | hi | n\_idx | replace | reward\_fn | reward | n\_games | n\_print | delay | curr\_epoch | curr\_params | lo\_eval | hi\_eval | n\_idx\_eval | replace\_eval | reward\_fn\_eval | reward\_eval | n\_games\_eval | n\_print\_eval | delay\_eval |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **10000000** | 10000 | 0 | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **10000000** | 10000 | 0 | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **50000000** | 10000 | 0 | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.01 | 0.00001 | 10000 | 0.9 | 0.1 | 0.00001 | 0 | False | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **50000000** | 10000 | 0 | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |

#### Monte Carlo Results

| gamma | epsilon | eps\_decay | s\_cost | q\_key\_fn | q\_key\_params | v\_fn | lo | hi | n\_idx | replace | reward\_fn | reward | n\_episodes | curr\_epoch | curr\_params | lo\_eval | hi\_eval | n\_idx\_eval | replace\_eval | reward\_fn\_eval | reward\_eval | n\_games\_eval | n\_print\_eval | delay\_eval |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.9 | 0.1 | 0.00001 | 0 | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **1000000** | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.9 | 0.1 | 0.00001 | 0 | **bin** | 2\_2 | vMax | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **1000000** | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.9 | 0.1 | 0.00001 | 0 | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **scalar** | 10\_10 | **100000** | 70000000 | 0\_0\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
| 0.9 | 0.1 | 0.00001 | 0 | **seq** | 2 | vSeq | 1 | 10000 | 50 | False | **topN** | 10\_10\_3 | **100000** | 70000000 | 0\_0\_10\_- | 1 | 10000 | 50 | False | scalar | 10\_1 | 10000 | 1000 | 0 |
