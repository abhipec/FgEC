# FgEC

Updated version of abhipec/fnet code, compatible with TensorFlow 1.10
Transfer learning part of abhipec/fnet is not included in this code.


## Publication
[Fine-Grained Entity Type Classification by Jointly Learning Representations and Label Embeddings. Abhishek, Ashish Anand and Amit Awekar. EACL 2017.](http://www.aclweb.org/anthology/E/E17/E17-1075.pdf)

Please use the following BibTex code for citing this work.

```
@InProceedings{abhishek-anand-awekar:2017:EACLlong,
  author    = {Abhishek, Abhishek  and  Anand, Ashish  and  Awekar, Amit},
  title     = {Fine-Grained Entity Type Classification by Jointly Learning Representations and Label Embeddings},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {797--807},
  url       = {http://www.aclweb.org/anthology/E17-1075}
}

```

## Instruction to use code

### Install dependencies
Python version: 3.6
```
pip install tensorflow-gpu scipy docopt joblib
```

### Download the glove word embeddings
Download the glove word embedding: http://nlp.stanford.edu/data/glove.840B.300d.zip
and store the file at location FgEC/data/glove.840B.300d.txt

### Compile the cpp files

cd FgEC/lib/
bash compile_gcc_5.bash

### Training
A sample file to train on OntoNotes dataset is available at FgEC/src/scripts/ontonotes.bash 

Please refer that file for further instructions to run the code.

