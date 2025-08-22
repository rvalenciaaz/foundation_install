# foundation_install
How to install ESM2 and other models locally via Hugging Face (HF) 

1) First, check that cuda-toolkit and nvcc are installed system-wide. Use cuda-toolkit version >=12.6.

**For Evo1:**

2) Create a environment using mamba:

```
mamba create -n model_env python==3.10
mamba activate model_env
```

(it can be a Python >=3.10 and <=3.12)

3) Then, install pytorch, transformers, and hf_transfer (for faster model tensors download):

```
pip3 install torch torchvision torchaudio
pip3 install transformers huggingface_hub hf_transfer
```

4) For Evo1, two libraries are important:
```
mamba install libxcrypt
pip install --upgrade flash-attn==2.7.4.post1 --no-build-isolation
```
5) Download the model folder (from togetherAI) using

```
bash download_hf_evo.sh
```
(You can change the version inside the bash sript to download a lighter version) 

6) There are two parts that might be tricky:
   - First, there are py libraries in the model folder that are needed for initialising the model. These can be symlinked or hardcopied to the corresponding include folder:
  
     ```  
     cp ~/workingfolder/models/evo-1-131k-base/*.py  ~/.cache/huggingface/modules/transformers_modules/evo-1-131k-base/
     ```
   - Second, it may happen that when initialising the model using the python script, it doesn't recognise crypt.h as available in the include folder. It is caused by a slightly different include folder being used in the call. To fix it, just symlink or hardcopy crypt.h to the right folder:

     ```  
     ln -s /home/yourusername/.local/share/mamba/envs/esm2_env/include/crypt.h /home/yourusername/.local/share/mamba/envs/esm2_env/include/python3.x/
     ```
    (x should be your version)
7) Execute the python script with the embedding pipeline and see if the output `embeddings.tsv` is correct:

   ```
   python evo_embed_hf_offline.py
   ```

**For ESM2:**

Perform the same steps until 3, and then you can skip 4 (you can use the same environment and it should be fine)

5) Download the model folder (from facebook) using

```
bash download_hf_esm2.sh
```
(You can change the version inside the bash sript to download a lighter version) 

6) Execute the python script with the embedding pipeline and see if the output `embeddings.tsv` is correct:

```
python esm2_embed_hf_offline.py
```

**Finetuning ESM2 for sequence classification using Transformers**


```
pip3 install evaluate accelerate peft bitsandbytes
```

