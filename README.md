# Multimodal Prompting

*Note: Supporting article available [here](https://nihaljn.github.io/posts/multimodal-prompting/)*.

This repository presents yet another extension to the Stable Diffusion model for image generation: a simple and fast strategy towards prompting Stable Diffusion with an arbitrary composition of image and text tokens.

<img src="assets/tiger.png">

## Usage

1. Clone this repository. This would also clone the original [Stable Diffusion repository](https://github.com/CompVis/stable-diffusion/) as a submodule. Before proceeding to the next steps, make sure to follow the instructions to set up text-to-image generation inside the `stable-diffusion` folder by following the steps from their repository. Note that this includes downloading their model into the appropriate place inside `stable-diffusion` and installing the requirements of `stable-diffusion`.

2. This repository contains the n-gram prompts data inside `data`. Additionally, the preprocessed CLIP embeddings of these prompts need to be downloaded into `data` from [here](https://drive.google.com/file/d/1yfudLHS7cDdixQ25xbQY55LohpB6Fl01/view?usp=sharing).

3. Design a multimodal prompt by replacing images with `[img]` token in the prompt string. For example, the prompt in the Figure above is: `prompt = "A tiger taking a walk on [img]`".

4. Let the comma separated paths to all images in the prompt be `image_paths`. Each image can optionally be assigned absolute weights (denoted as $m$, see [this](https://nihaljn.github.io/posts/multimodal-prompting/) for more details) ideally in the integer range 1-10. Let the comma separated weights be `image_weights`.

5. Run the following command:

```bash
python mp2img.py \
    --prompt $prompt \
    --plms \
    --prompt-images $image_paths \
    --image-weights $image_weights
```

Some sample commands are shown in `commands.sh`.

## More Results

<img src="assets/cyber1.png">

<img src="assets/cyber2.png">

<img src="assets/cyber3.png">

We see the effect of varying $m$: weighting different images in the prompt reflects the corresponding details in the generation while staying true to the desired weighting.
