# foundation_install
How to install ESM2 and other models locally via HF 

1) First, check that cuda-toolkit and nvcc are installed system-wide. Use cuda-toolkit version >=12.6.

**For Evo1:**

2) Create a environment using mamba:

```
mamba create -n model_env python==3.10
```

(it can be a Python >=3.10 and <=3.12)

3) Then, install pytorch, transformers, and hf_transfer (for faster model tensors download):

```
pip3 install torch torchvision torchaudio
pip3 install transformers
pip3 install hf_transfer
```

4) For Evo1, two libraries are important:
```
mamba install libxcrypt
pip install --upgrade flash-attn==2.7.4.post1 --no-build-isolation
```
5) Download the model folder using

bash 

6) There are two parts that might be tricky:
   - First, there are py libraries in the model folder that are needed for initialising the model  

   - Second, it may happen that when initialising the model using the python script, it doesn't recognise crypt.h as available in the include folder. It is caused by a slightly different include folder is used in the calling. To fix it, just symlink crypt.h to the right folder:
