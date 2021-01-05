import argparse
import json
import torch
import subprocess
from datetime import datetime
from data import SentencePieceVocabulary

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='rnng.pt', required=True)

DEFAULT_SPEC = {
  "name": "rnng-pytorch",
  "ref_url": "https://github.com/hiroshinoji/rnng-pytorch",
  "image": {
    "maintainer": "h.nouji@gmail.com",
    "version": "<image.sha1>",
    "checksum": "<image.files_sha1>",
    "datetime": "<image.datetime>",
    "gpu": {
      "required": True,
      "supported": True
    },
    "supported_features": {
      "tokenize": True,
      "unkify": True,
      "get_surprisals": True,
      "get_predictions": False,
      "mount_checkpoint": True
    }
  },
  "vocabulary": {
    "unk_types": ["<unk>"],
    "prefix_types": [],
    "suffix_types": [],
    "special_types": []
  },
  "tokenizer": {
    "type": "word",
    "cased": True
  }
}

def mk_spec(model_path, vocab):
  j = DEFAULT_SPEC
  j["image"]["checksum"] = (subprocess.check_output("cat {} | sha1sum".format(model_path), shell=True)
                            .decode('utf-8').split()[0])
  j["image"]["version"] = (subprocess.check_output("git rev-parse --verify HEAD", shell=True)
                           .decode('utf-8').strip())
  j["image"]["datetime"] = str(datetime.now())
  if isinstance(vocab, SentencePieceVocabulary):
    j["vocabulary"]["unk_types"] = [vocab.unktoken]
    j["is_subword"] = True
  else:
    j["vocabulary"]["unk_types"] = vocab.specials
    j["is_subword"] = False
    assert '<unk>' in j["vocabulary"]["unk_types"]
    j["vocabulary"]["items"] = [w for w in vocab.i2w if w not in vocab.specials]
  return j

def main(args):
  checkpoint = torch.load(args.model)
  vocab = checkpoint['vocab']
  spec = mk_spec(args.model, vocab)
  print(json.dumps(spec))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

