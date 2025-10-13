from datasets import load_dataset
import os
import json
import math

batch_size = 10_000
dataset = load_dataset('cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois', split='train')
n_batches = math.ceil(len(dataset) / batch_size) 

print(f'batch size {batch_size} || num batches {n_batches}')

for i, batch in enumerate(dataset.iter(batch_size=batch_size), 1):
    input_json = [{'arxiv_id': x} for x in batch['arxiv_id']]
    json.dump(input_json, open('tmp.json', 'w'), indent=2)
    print(f'[batch {i} / {n_batches}] wrote json, starting download')

    os.system('python kaggle_arxiv_dataset_dl.py -i tmp.json -o arxiv_pdfs')
    print(f'[batch {i} / {n_batches}] finished download')
