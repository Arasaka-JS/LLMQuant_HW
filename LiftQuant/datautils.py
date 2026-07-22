import pdb
import os
from transformers import AutoTokenizer
from datasets import DownloadConfig, load_dataset
import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_dataset_cache_dir(name, cache_dir=None):
    env_var = f"{name.upper()}_CACHE_DIR"
    env_cache_dir = os.environ.get(env_var)
    if env_cache_dir and os.path.exists(env_cache_dir):
        return env_cache_dir, True

    base_cache_dir = cache_dir or os.environ.get("HF_DATASETS_CACHE") or "./cache"
    cache_path = os.path.join(base_cache_dir, "datasets", name)
    cache_exists = os.path.exists(cache_path)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path, cache_exists


def get_download_config(use_local_files):
    if use_local_files:
        return DownloadConfig(local_files_only=True)
    return None

def get_redpajama(nsamples, seed, seqlen, model): 
    print("get_redpajama") 
    #traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir = "../dataset/RedPajama-1T-Sample",split='train')  
    redpajama_cache_dir = os.environ.get("REDPAJAMA_CACHE_DIR")
    if redpajama_cache_dir and not os.path.exists(redpajama_cache_dir):
        raise FileNotFoundError(f"REDPAJAMA_CACHE_DIR does not exist: {redpajama_cache_dir}")
    dataset_cache_dir, use_local_files = get_dataset_cache_dir(
        "redpajama",
        redpajama_cache_dir,
    )
    download_config = get_download_config(use_local_files)
    print(f"RedPajama cache: {dataset_cache_dir} (local_files_only={use_local_files})")
    traindata = load_dataset(
        "ZengXiangyu/RedPajama-Data-1T-Sample",
        cache_dir=dataset_cache_dir,
        split='train',
        download_config=download_config,
    )  
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed) 
    trainloader = []
    
    for _ in range(nsamples):
        while True:
            i = random.randint(0, int(len(traindata)) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='../val.jsonl.zst', split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, seed, seqlen, model, cache_dir=None):
    print("get_wikitext2")
    dataset_cache_dir, use_local_files = get_dataset_cache_dir("wikitext2", cache_dir)
    download_config = get_download_config(use_local_files)
    
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir, download_config=download_config)
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir, download_config=download_config)
    print("get_wikitext2 over")


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt' )
    

    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, cache_dir=None):
    print("get_c4")
    dataset_cache_dir, use_local_files = get_dataset_cache_dir("c4", cache_dir)
    download_config = get_download_config(use_local_files)
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=dataset_cache_dir, download_config=download_config
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=dataset_cache_dir, download_config=download_config
    )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    print("get_c4_new")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', cache_dir=None,
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, cache_dir)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)  
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)  
        return get_c4(nsamples, seed, seqlen, model, cache_dir)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model)
        train=wiki_train+ptb_train+c4_train
        val=None
        return train,val
