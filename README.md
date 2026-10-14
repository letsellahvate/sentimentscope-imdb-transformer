# SentimentScope: Movie Review Sentiment Analysis with a Transformer

A transformer model built from scratch in PyTorch that reads an IMDB movie review and decides whether it is **positive** or **negative**. Built for the Udacity *AWS AI Programmer* nanodegree.

**Result: 76.58% accuracy on 25,000 unseen test reviews** (target: >75%).

## What's in this repo

| File | What it is |
|---|---|
| `SentimentScope_starter.ipynb` | The full notebook: data loading, exploration, model, training and evaluation |
| `sentiment_model.pt` | The trained model weights (PyTorch checkpoint) |

## How it works

1. **Data** – 50,000 IMDB reviews (half positive, half negative). The 25,000 training reviews were split 90/10 into training (22,500) and validation (2,500) sets.
2. **Tokenization** – each review is split into sub-word pieces with the `bert-base-uncased` tokenizer and cut or padded to 128 tokens.
3. **Model (`DemoGPT`)** – token + position embeddings → stacked transformer blocks → layer norm → **mean pooling** (average all tokens into one vector) → linear layer that outputs a score for each of the 2 classes.
4. **Training** – AdamW optimizer (learning rate 3e-4), cross-entropy loss, 6 epochs on a GPU.

## Results

| Stage | Accuracy |
|---|---|
| Untrained model (random guessing) | ~47% |
| Best validation accuracy (epoch 4) | 79.08% |
| **Final test accuracy** | **76.58%** |

## Key takeaways

- **Mean pooling turns a text-generation model into a classifier.** Averaging every token into one vector lets the model make a single call about the whole review.
- **More training helps, up to a point.** Going from 3 to 6 epochs raised test accuracy from 74.2% to 76.6%, but validation accuracy peaked at epoch 4 while training loss kept falling: a sign of overfitting.
- **Reading only 128 tokens limits accuracy.** The average review is ~230 words, so the model often misses the ending, where the verdict usually is. A longer context, a bigger model, or fine-tuning a pre-trained model like BERT should push accuracy to 85–90%.

## Run it yourself

```bash
pip install torch transformers pandas matplotlib scikit-learn
```

Download the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/), extract it next to the notebook (so you have an `aclImdb/` folder), and run the notebook top to bottom. A GPU is strongly recommended.

To load the trained model, run the notebook's setup and model-definition cells, then:

```python
model = DemoGPT(config)
model.load_state_dict(torch.load("sentiment_model.pt", map_location="cpu"))
model.eval()
```
