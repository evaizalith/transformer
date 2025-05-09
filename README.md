# transformer

This is a pytorch implementation of a transformer with a baseline feedforward model as a comparison.

## Usage

1. Install all requirements from the `requirements.txt` file.

2. Place a txt file in the root directory containing your training data.

3. Run `python src/main.py` 

## Architecture

![transformer drawio](https://github.com/user-attachments/assets/93fbc813-b6df-4dc9-9143-c7fe083a50f2)

## Results

The transformer was compared to the baseline model using the Wikitext dataset. 

![transformer_homework_loss](https://github.com/user-attachments/assets/ae7e7561-e462-418e-a3ce-8093b55d2227)

The transformer had an inference time of 125.07 ms, whereas the baseline model had an inference time of 82.55 ms. Each had 1,350,273 and 545,153 parameters respectively. 

Overall, the baseline model struggled. It simply could not learn the structure of the linguistic data presented to it, and thus failed to converge. On the other hand, the transformer, though its loss is still high at the end of its limited training period, is clearly converging, really learning the structure of language. 


