# Finetuning LLama3 on the ARC Prize


The [ARC Prize](https://arcprize.org/) has recently started a Kaggle competition. Previously the contest was dominated by "DSL" algorithms, but this time around it seems like LLMs might take the cake.

On public eval, Claude 3.5 and GPT-4o already achieve impressive results of 21% and 9% respectively at a Baseline.

![](Pasted image 20240722164652.png)
(22.07.24 - https://arcprize.org/leaderboard ARC-AGI-Pub)


So it seems prudent to check how far simple LLM approaches can go. 
We will try to do this *cheaply* and *quickly*, which is why we will finetune LLama3-8B using a LoRA, achieving $12.75\%$ accuracy on the public eval task, beating GPT-4o using less than $10 in compute cost.

![](README_files/montage.jpg)
##### Hardware

Finetuning LLMs, even with LoRA adapters can have high hardware requirements, often needing at least 20GB of VRAM. While this is a lot for consumer hardware, renting GPUs is cheap, so we can test our approach using a A10 GPU and a A100 GPU on the [Lambda GPU cloud](https://lambdalabs.com/service/gpu-cloud)

The whole training and inference pipeline runs for ~6h on A10 and ~3.5h on A100 costing around $6 and $7.50 respectively.



#### The Problems


ARC Problems have a visual format, which introduces additional complexity for Language Models, as they cannot easily reason over 2D data. To mitigate this we create additional ARC problems:

For each challenge we create 7 additional challenges, corresponding to rotation and transposition+rotation of the original problem. 

I.e. if this is a valid challenge:
![](Pasted image 20240723105247.png|500)
Then so is this:
![](Pasted image 20240723105234.png|500)

This not only helps the models to learn patterns in vertical directions, but also increases our training data size eight-fold!

---

We convert each of these problems into a simple text format, making sure that there are no tokenization issues. This is done by simply representing the matrix as a string and adding linebreaks for each new row. We additionally specify if a given matrix is an input or an output, and clearly mark the beginning and end of each example.

![](arc 1.png)
#### Training

Using only the input-output examples for each Problem, we then finetune our model using 1-2 epochs on the collected data. To save some computation time and VRAM we use the unsloth library, which is easy to install:

---
##### Setup

Create and activate a virtual environment:
```bash
python -m venv --system-site-packages .unsloth
source .unsloth/bin/activate
```

Update torch and triton:
```bash
pip install --upgrade torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121
```

Install unsloth:
```bash
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
```

and we are ready to go.


##### Setting up LoRA

We use a LoRA model, with the following parameters:

```python
#create PEFT LoRA model
model = FastLanguageModel.get_peft_model(
	model,
	r = 16, #Rank of the LoRA weights
	target_modules = ["q_proj", "k_proj", 
					"v_proj", "o_proj",
					"gate_proj", "up_proj", "down_proj",
					"embed_tokens", "lm_head",], #which modules we want to target
	lora_alpha = 16,
	lora_dropout = 0, # Supports any, but = 0 is optimized
	bias = "none",    # Supports any, but = "none" is optimized
	use_gradient_checkpointing = "unsloth", 
	random_state = 1212,
	use_rslora = True,  # Rank stabilized Lora
	loftq_config = None, 
)
```

and then we simply start training. 

![](Pasted image 20240724090235.png)

#### Prediction

The ARC contest allows up to two predictions for each challenge. As noted before, LLMs do not reason well over 2D, as they use the 1D text representation. For this reason, we create our two guesses by predicting the normal challenge and the transposed challenge. This should help the LLM to solve problems that require a lot of vertical argumentation. 

We produce results for these tasks, and simply ignore predictions with generation errors. We use top-k inference with $k=50$. In the end, we produce guesses for 339 of the 400 challenges.
In these 339 guesses there are 51 correctly solved Problems! This corresponds to a score of $12.75\%$


| Num Problems | Predictions | Solved by any Guess | Solved by first Guess | Solved by second Guess |
| ------------ | ----------- | ------------------- | --------------------- | ---------------------- |
| 400          | 339         | 51                  | 38                    | 28                     |

Our finetuned LLama3 models with some data augmentation beats the GPT-4o Baseline on the public eval dataset.

##### Correct Prediction Examples

![](Pasted image 20240723102108.png)
>"Restore the symmetric pattern"
   I am surprised the LLM managed to solve this correctly. The action needed is not complex, but learning that action seems unlikely. The transposed prediction repeated a previous output.

![](Pasted image 20240723102056.png)
>"Extend the pattern, repeating the colors"
   This is probably very easy, at the LLM can simply use a train output and remap the colors.

![](Pasted image 20240723102041.png)
> "Repeat the lines marked with grey"

![](Pasted image 20240723102012.png)
> This is an overlay task, Brown>Yellow>Blue>Grey. 

![](Pasted image 20240723101932.png)
> "Draw line between yellow points, permuting colors of cells between them" 
   Here we see that the transposed version has an easier time, as it has to draw lines from left to right instead of up to down.

![](Pasted image 20240723101908.png)
> "Shift the pixels based on color"
   This is probably rather easy to learn for the LLM. The transposed version struggles, as it has to do the same task but from top to bottom instead of left to right

---


##### Common Failures of LLama and Conclusion

The LLama models often chose to repeat the last train output, instead of predicting a new guess. This might be caused by overfitting during training and can perhaps be mitigated by shuffling the input-output examples between epochs. Another issues was that LLama could not predict challenges that had a big input and output size, quickly running into token limits.

At the same time the base accuracy is already impressive, and LLama very rarely generated solutions that were formatted wrongly. The problems it can solve are not simple, and in the challenges it cannot solve, we have several near misses. It seems to work particularly well when the solution has repeating patterns or uses bite-wise operations of the input.

Using Lambda GPUs this training cost less than x$ and finished in y time, showing that small models can rather cheaply be made competitive if finetuned well.




#### Costs and time taken

|                    | A10               | A100              | H100             |
| ------------------ | ----------------- | ----------------- | ---------------- |
| Training Time      | 21207s (5h53m27s) | 12422s (3h27m02s) | 8243s (2h17m23s) |
| Inference Time     | 7371s             | 8550s             | 7688s            |
| Hourly Cost        | $0.75             | $1.29             | $2.49            |
| Estimated Cost     | $5.95             | $7.50             | $11.02           |
| Training only Cost | $4.42             | $4.45             | $5.70            |



#### Appendix: Some Interesting Errors

Errors can be divided into 4 categories
1. No solution generated/Generation could not be parsed
2. Repetition of known seen training output
3. Misunderstanding of Problem
4. Near misses

Here we show some interesting cherry picked examples of the latter two categories:

![](21_60a26a3e.png)
>Slight misunderstanding of task


![](21_84db8fc4.png)
> *Almost* correct prediction

![](21_93b4f4b3.png)
> Failure to fit shapes, probably not understanding task


![](21_137f0df0.png)
> Understands that inside of grey shape should be filled with red, does *almost* manage to add blue outside, but not quite

![](21_319f2597.png)
> The non-transposed version manages to correctly draw the horizontal line, the vertical line does not work at all

![](21_516b51b7.png)
> Here the non-transposed prediction repeats a previous output, the transposed prediction almost understands the task, but fails to apply it completely


![](21_0692e18c.png)
> Complete misunderstanding of task, but color is predicted correctly.




