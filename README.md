<h1 align="center"> Document of AI-Endo</h1>

[//]: # (<HR SIZE=10>)
<p align="justify">This is the pytorch implementation of paper "<i>Intelligent Surgical Workflow Recognition for Endoscopic
Submucosal Dissection with Real-time Animal Study</i>" by Jianfeng Cao, Hon-Chi Yip, Yueyao Chen, Markus Scheppach, Xiaobei Luo,
Hongzheng Yang, Ming Kit Cheng, Yonghao Long, Yueming Jin, Philip Wai-Yan Chiu, Yeung Yam, Helen Mei-Ling Meng, and Qi Dou.</p>

<h2>Dependency installation</h2>
The model is developed based on pytorch. To install dependencies, run

<pre>
git clone https://github.com/med-air/AI-Endo.git
cd AI-Endo
conda env create -f environment.yml
conda activate AI-Endo
</pre>

<h2>Data preparation</h2>
AI-Endo is trained with downsampled images of endoscopic video. The zip files are located at `/home/ryukijano/AI-Endo/23506866`. To prepare the data, extract the zip files and arrange them as follows:

<pre>
/home/ryukijano/AI-Endo/23506866--|
                             |--Images--|
                             |          |--Video1--|
                             |          |          |--Image00001.png
                             |          |          |--Image00002.png
                             |          |...
                             |
                             |--Labels--|--Phase1.txt
                                        |--Phase2.txt
                                        |...
</pre>
<code>DATA_ROOT</code> should be set to `/home/ryukijano/AI-Endo/23506866` in the config files, e.g., <code>configs/test.yml</code>, accordingly.

<h2>Train</h2>
<p align="justify">The training process of AI-Endo includes two stages, ResNet50 and Fusion+Transformer. To execute the
training process, the dataset should be specified in the config file <code>./configs/train.yml</code>, such as paths of downsampled 
video at 1 fps and its corresponding annotations.</p>

<pre>
python get_paths_labels.py
python train_all.py --cfg train
</pre>

<h2>Prediction</h2>
<p align="justify">Set the file paths of trained models in <code>./configs/test.yml</code> and run</p>

<pre>
# Option 1: offline prediction
python test_all.py --cfg test_offline

# Option 2: online prediction
python online.py -s --cfg test
</pre>
Pretrained mdoels are available at [Google Drive](https://drive.google.com/drive/folders/1aMgEuxhZjLtSJ3ica6EVKYkGeMGG1Vtw?usp=share_link).

<h2>Acknowledgment</h2>
The code of this repository is partially referred to <a href="https://github.com/xjgaocs/Trans-SVNet">Trans-SVNet</a> and <a href="https://github.com/YuemingJin/TMRNet">TMRNet</a>.

<h2>Citation</h2>
TBD

<h2>Correspondence</h2>
<p align="justify">For further question about the code, please contact <code>jianfeng13.cao@gmail.com</code>.</p>

<h2>LICENSE</h2>
<p align="justify">This project is covered under the MIT License.</p>
