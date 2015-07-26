# cifar.torch

The code achieves 92.45% accuracy on CIFAR-10 just with horizontal reflections.

Data preprocessing:

```bash
th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
Takes about 40 minutes and saves 1400 Mb file.

Training:

```bash
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg
```
