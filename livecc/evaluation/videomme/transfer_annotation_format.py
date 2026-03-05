import json

if __name__ == '__main__':
    path = '/2022233235/.cache/huggingface/hub/video-mme/val.json'
    save_path = 'videomme.jsonl'
    annos = []
    datums = json.load(open(path))
    for datum in datums:
        annos.append(datum)
    with open(save_path, 'w') as f:
        for anno in annos:
            f.write(json.dumps(anno) + '\n')