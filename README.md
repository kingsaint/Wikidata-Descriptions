# Wikidata-Descriptions
Be Concise and Precise: Synthesizing Open-Domain Entity Descriptions from Facts

## Training and Inference
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
python evaluate.py <model_name>.csv
```

## Authors

* **Rajarshi Bhowmik**  - [website](https://kingsaint.github.io)
* **Gerard de Melo** - [website](http://gerard.demelo.org)

## Citation

If you use this code, please cite our paper.

```
@inproceedings{BhowmikDeMelo2019EntityDescriptionsCopyModel,
  title = {Be Concise and Precise: Synthesizing Open-Domain Entity Descriptions from Facts},
  author = {Bhowmik, Rajarshi and {de Melo}, Gerard},
  booktitle = {Proceedings of The Web Conference 2019},
  year = {2019},
  location = {San Francisco},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
