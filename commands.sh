# python mp2img.py \
#     --prompt "[img1] in the style of [img2]" \
#     --plms \
#     --prompt-images sample_images/garden.png,sample_images/cyberpunk.webp \
#     --image-weights 4,4

# python mp2img.py \
#     --prompt "[img1] in the style of [img2]" \
#     --plms \
#     --prompt-images sample_images/garden.png,sample_images/cyberpunk.webp \
#     --image-weights 2,4

# python mp2img.py \
#     --prompt "[img1] in the style of [img2]" \
#     --plms \
#     --prompt-images sample_images/garden.png,sample_images/cyberpunk.webp \
#     --image-weights 4,2

python mp2img.py \
    --prompt "[img1] in the style of [img2]" \
    --plms \
    --prompt-images sample_images/garden.png,sample_images/cyberpunk.webp \
    --image-weights 4,2

python mp2img.py --prompt "albert einstein looking like [img]" --plms --prompt-images sample_images/andy_warhol.webp --image-weights 4