# Wikidata-Descriptions
Be Concise and Precise: Synthesizing Open-Domain Entity Descriptions from Facts

# Training and Inference
```
run_model.py --model MODEL_NAME --datadir PATH_TO_DATADIR --config PATH_TO_CONFIG_FILE_NAME --mode [training/eval]
```
e.g.
```
run_model.py --model NKLM --datadir Data/WikiFacts10k-OpenDomain --config Configs/NKLM-config.json --mode training
```
## Evaluation

To run evaluation script, use

```
python evaluate.py result_<model_name>.csv
```

## Authors

* **Rajarshi Bhowmik**  - [website](https://kingsaint.github.io)
* **Gerard de Melo** - [website](http://gerard.demelo.org)

## Citation

If you use this code, please cite our paper.

```
@inproceedings{Bhowmik2018EntityDescriptions,
  title = {Generating Fine-Grained Open Vocabulary Entity Type Descriptions},
  author = {Bhowmik, Rajarshi and {de Melo}, Gerard},
  booktitle = {Proceedings of ACL 2018},
  year = {2018},
  location = {Melbourne},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
