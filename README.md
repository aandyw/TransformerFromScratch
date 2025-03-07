# TransformerFromScratch
🤖

## Setup

Setup the local development environment using

```bash
./scripts/setup_env.sh     # Create virtual env & download dependencies
source .venv/bin/activate  # Activate it
```

## Exercise
- `notebooks/transformer_from_scratch_exercise.ipynb` contains a practice notebook to build a transformer from scratch by filling in missing portions of the code. It's a good way to review your knowledge of Transformers.

- The solution is in `notebooks/transformer_from_scratch_solution.ipynb`.

## Training
*Training for the GPU poor T_T*
1. Upload the `notebooks/train_model.ipynb` notebook into colab (or kaggle) and run on GPU

*If you're GPU self-sufficient you can run locally:*
1. Login to huggingface with `huggingface-cli login`
2. Run `python transformer/train.py`


# Acknowledgements
A variety of resources that really helped us out in understanding and implementing the Transformer model

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference](https://www.youtube.com/watch?v=ISNdQcPhsts) by Umar Jamil
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) by 3Blue1Brown
- [Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) by Sebastian Raschka
- [Self Attention in Transformer Neural Networks (with Code!)](https://www.youtube.com/watch?v=QCJQG4DuHT0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4) by CodeEmporium
- Visualizing attention matrix using [BertViz](https://github.com/jessevig/bertviz)