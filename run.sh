dataset=Amazon-531 # DBPedia-298
gpu=0


# Section 3.1: LLM-Enhanced Core Class Annotation

# LLM-Based Taxonomy Enrichment (need OpenAI API)
# output: train/llm_enrichment.json
python3 LLM_taxonomy_enrichment.py --data_dir data/ --dataset ${dataset}

# Initial core class annotation (need OpenAI API)
# input: llm enrichment
# output: train/init_core_classes.json
python3 core_class_annotation.py --data_dir data/ --dataset ${dataset} --gpu ${gpu}



# Section 3.2: Corpus-Based Taxonomy Enrichment
# input: initial core classes, quality phrases and phrasal segmented corpus (see https://github.com/shangjingbo1226/AutoPhrase)
# output: /train/enrichment.txt
python3 taxonomy_enrichment.py --data_dir data/ --dataset ${dataset} --gpu ${gpu}


# Section 3.3: Core Class Refinement
# input: initial core classes, enriched taxonomy
# output: train/refined_core_classes.json
python3 core_class_refinement.py --data_dir data/ --dataset ${dataset} --gpu ${gpu}


# Section 3.4: Text Classifier Training with Path-Based Data Augmentation

# Path-based generation (need OpenAI API)
# output: train/generated_docs.txt & train/generated_doc2label.json
python3 generation.py --data_dir data/ --dataset ${dataset}


# Text classifier training
# input: refined core classes and generated documents
# output: trained classifier at: train/model.pt
python3 prepare_training_data.py --data_dir data/ --dataset ${dataset} --gpu ${gpu}
python3 classifier_training.py --data_dir data/ --dataset ${dataset} --gpu ${gpu}


# Evaluation, if a test set is available
# need: test/corpus.txt & test/doc2labels.txt
python3 eval_classifier.py --data_dir data/ --dataset ${dataset} --gpu ${gpu} --model_pth data/${dataset}/train/model.pt