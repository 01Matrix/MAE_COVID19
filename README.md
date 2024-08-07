# Pre-trained Natural Image Models are Few-shot Learners for Medical Image Classification: COVID-19 Diagnosis as an Example

This repository contains the code and datasets for the paper "Pre-trained Natural Image Models are Few-shot Learners for Medical Image Classification: COVID-19 Diagnosis as an Example". It hleps to direct users to reproduce our reported results.


## Contents

- [x] Visualization demo
- [x] Dataset stats
- [x] Pre-trained models + Intermediate models + fine-tuning code
- [x] Pre-training code

## Visualization Demo:
<style>
  .image-container {
    text-align: center;
    margin: 0 auto;
  }
  .image-container img {
    width: 100%;
    max-width: 600px;
  }
  figcaption {
    font-style: 14px;
    margin-top: 6px; 
  }
</style>
<figure class="image-container">
  <img src="./demo/C14_pretrain_model.png" alt="C14_pretrain_model">
  <figcaption>Visualization of MAE-B/16_DATA14</figcaption>
</figure>

<figure class="image-container">
  <img src="./demo/C1000_pretrain_model.png" alt="C1000_pretrain_model">
  <figcaption>Visualization of MAE-B/16_C1000</figcaption>
</figure>

<!-- (1) C14_pretrain_model:
<p align="left">
  <img src="./demo/C14_pretrain_model.png" width="600" title="C14_pretrain_model">
  <figcaption>这是图片的标题</figcaption>
</p>

(2) C1000_pretrain_model:
<p align="left">
  <img src="./demo/C1000_pretrain_model.png" width="600" title="C1000_pretrain_model">
</p> -->

## Dataset for pre-training

<table>
<caption>Specifications of the 14 self-collected public COVID-19 CT image datasets making up of the pre-training composite dataset</caption>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Abbreviation</th>
<th valign="bottom">COVID</th>
<th valign="bottom">Normal</th>
<th valign="bottom">Bacteria</th>
<th valign="bottom">Dataset size</th>

<tr><td align="left">chest-ct-scans-with-COVID-19</td> <td align="left"> CHE </td><td align="left"> 27,781 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left">  27,781 </td></tr>
<tr><td align="left">COVID-19_ct_scans1</td> <td align="left"> CCT </td><td align="left">  1,762 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left">  1,762 </td></tr>
<tr><td align="left">COVID-19-20_v2</td> <td align="left"> C1920 </td><td align="left">  6,723 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left">  6,723 </td></tr>
<tr><td align="left">COVID-19-AR</td> <td align="left"> CAR </td><td align="left">  18,592 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left">  18,592 </td></tr>
<tr><td align="left">COVID-19-CT-segmentation-dataset</td> <td align="left"> CCS </td><td align="left">  110 </td><td align="left"> 	0 </td><td align="left"> 	0 </td><td align="left"> 	110 </td></tr>
<tr><td align="left">COVID19-CT-Dataset1000+</td> <td align="left"> C1000 </td><td align="left">  307,765 </td><td align="left"> 	0 </td><td align="left"> 	0 </td><td align="left"> 	307,765 </td></tr>
<tr><td align="left">CT_Images_in_COVID-19</td> </td><td align="left"> 	CIC	</td><td align="left">  32,548 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left">  32,548 </td></tr>
<tr><td align="left">MIDRC-RICORD-1A</td> <td align="left"> MRA </td><td align="left">  9,833 </td><td align="left">  0 </td><td align="left">  0 </td><td align="left"> 	9,833 </td></tr>
<tr><td align="left">MIDRC-RICORD-1B</td> <td align="left"> MRB </td><td align="left">  5,501 </td><td align="left"> 	0 </td><td align="left"> 	0 </td><td align="left"> 	5,501 </td></tr>
<tr><td align="left">sarscov2-ctscan-dataset</td> </td><td align="left">  SC </td><td align="left">  1,252 </td><td align="left">  1,229 </td><td align="left">  0 </td><td align="left">  2,481 </td></tr>
<tr><td align="left">SIRM-COVID-19</td> <td align="left"> SIRM </td><td align="left"> 	599 </td><td align="left">  0	</td><td align="left">  0	</td><td align="left">  599 </td></tr>
<tr><td align="left">COVIDX-CT-2A</td> <td align="left"> CXC </td><td align="left">  93,975	</td><td align="left">  59,510 </td><td align="left">  0 </td><td align="left">  153,485 </td></tr>
<tr><td align="left">large-COVID-19-ct-slice-dataset</td> <td align="left"> LC </td><td align="left">  7,593 </td><td align="left">  6,893 </td><td align="left">  0 </td><td align="left">  14,486 </td></tr>
<tr><td align="left">COVID-19-and-common-pneumonia-chest-CT-dataset</td> <td align="left"> CC </td><td align="left">  41,813 </td><td align="left">  0 </td><td align="left">  55,219 </td><td align="left">  97,032 </td></tr>
<tr><td align="left"> Summation </td> <td align="left">  / </td> <td align="left">  555,847 </td> <td align="left">  67,632 </td> <td align="left">  55,219 </td> <td align="left">  678,698 </td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<!-- TABLE BODY -->
</tbody>
</table>

## Pre-training recipes

The pre-training recipes are in [PRETRAIN.md](PRETRAIN.md).

## Fine-tuning recipes
### Pre-trained models for fine-tuning

<table>
<caption>The following table provides the pre-trained image models for fine-tuning. Supervised models are pre-trained with vanilla ViT, and self-supervised models are pre-trained with MAE</caption>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Training dataset</th>
<th valign="bottom">Training method</th>
<th valign="bottom">Image domain</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="">ViT-B/16_IN1K</a></td><td align="left">ViT-B/16</td><td align="left">IN1K</td><td align="left">SL</td><td align="left">Natural image</td></tr>
<tr><td align="left"><a href="">ViT-L/16_IN1K</a></td><td align="left">ViT-L/16</td><td align="left">IN1K</td><td align="left">SL</td><td align="left">Natural image</td></tr>
<tr><td align="left"><a href="">ViT-B/16_CXC</a></td><td align="left">ViT-B/16</td><td align="left">CXC</td><td align="left">SL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">ViT-L/16_CXC</a></td><td align="left">ViT-L/16</td><td align="left">CXC</td><td align="left">SL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-B/16_IN1K</a></td><td align="left">ViT-B/16</td><td align="left">IN1K</td><td align="left">SSL</td><td align="left">Natural image</td></tr>
<tr><td align="left"><a href="">MAE-L/16_IN1K</a></td><td align="left">ViT-L/16</td><td align="left">IN1K</td><td align="left">SSL</td><td align="left">Natural image</td></tr>
<tr><td align="left"><a href="">MAE-B/16_IN1K</a></td><td align="left">ViT-B/16</td><td align="left">CXC</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-L/16_IN1K</a></td><td align="left">ViT-L/16</td><td align="left">CXC</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA13</a></td><td align="left">ViT-B/16</td><td align="left">DATA13</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-L/16_DATA13</a></td><td align="left">ViT-L/16</td><td align="left">DATA13</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA14</a></td><td align="left">ViT-B/16</td><td align="left">DATA14</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td align="left"><a href="">MAE-L/16_DATA14</a></td><td align="left">ViT-L/16</td><td align="left">DATA14</td><td align="left">SSL</td><td align="left">Medical image</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

### Intermediate models for fine-tuning

<table>
<caption>The following table provides the intermediate models for fine-tuning. Supervised models are pre-trained with vanilla ViT, and self-supervised models are pre-trained with MAE</caption>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Intermediate model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Base model</th>
<th valign="bottom">Dataset for Phase-1</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="">ViT-B/16_IN1K/CXC</a></td><td align="left">ViT-B/16</td><td align="left">ViT-B/16_IN1K</td><td align="center">CXC</td></tr>
<tr><td align="left"><a href="">ViT-L/16_IN1K/CXC</a></td><td align="left">ViT-L/16</td><td align="left">ViT-L/16_IN1K</td><td align="center">CXC</td></tr>
<tr><td align="left"><a href="">MAE-B/16_IN1K/CXC</a></td><td align="left">ViT-B/16</td><td align="left">MAE-B/16_IN1K</td><td align="center">CXC</td></tr>
<tr><td align="left"><a href="">MAE-L/16_IN1K/CXC</a></td><td align="left">ViT-L/16</td><td align="left">MAE-L/16_IN1K</td><td align="center">CXC</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA13/CXC</a></td><td align="left">ViT-B/16</td><td align="left">MAE-B/16_DATA13</td><td align="center">CXC</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA13/CXC</a></td><td align="left">ViT-L/16</td><td align="left">MAE-L/16_DATA13</td><td align="center">CXC</td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

The fine-tuning recipes are in [FINETUNE.md](FINETUNE.md).

### Results

By fine-tuning the intermediate models, we achieve the best performance in few-shot real-world COVID-19 classification tasks from CT images (detailed in the paper):

<table>
<caption>The following table provides the results of full fine-tuning of intermediate models using U_orig across 12 different random seeds, with data split = 2:3:5. Intermediate models consistently demonstrate a remarkable performance gain compared to the corresponding pre-trained base models. Moreover, intermediate MAE models outperform intermediate ViT models significantly. Notably, MAE-L/16_IN1K/CXC performs the best among all the intermediate models</caption>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Intermediate model</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">F1</th>
<th valign="bottom">AUC</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="">ViT-B/16_IN1K/CXC</a></td><td align="left">U_orig</td><td align="left">0.7712 ± 0.0188</td><td align="left">0.7464 ± 0.0206</td><td>0.8456 ± 0.0114</td></tr>
<tr><td align="left"><a href="">ViT-L/16_IN1K/CXC</a></td><td align="left">U_orig</td><td align="left">0.7718 ± 0.0172</td><td align="left">0.7453 ± 0.0313</td><td>0.8494 ± 0.0159</td></tr>
<tr><td align="left"><a href="">MAE-B/16_IN1K/CXC</a></td><td align="left">U_orig</td><td align="left">0.8554 ± 0.0222</td><td align="left">0.8445 ± 0.0281</td><td>0.9337 ± 0.0113</td></tr>
<tr><td align="left"><a href="">MAE-L/16_IN1K/CXC</a></td><td align="left">U_orig</td><td align="left">0.8680 ± 0.0157</td><td align="left">0.8586 ± 0.0164</td><td>0.9380 ± 0.0125</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA13/CXC</a></td><td align="left">U_orig</td><td align="left">0.8434$ ± 0.0231</td><td align="left">0.8319 ± 0.0287</td><td>0.9258 ± 0.0117</td></tr>
<tr><td align="left"><a href="">MAE-B/16_DATA13/CXC</a></td><td align="left">U_orig</td><td align="left">0.8385$ ± 0.0255</td><td align="left">0.8355 ± 0.0242</td><td>0.9217 ± 0.0111</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

## License

* This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgement
* This repo is adapted from the [MAE](https://github.com/facebookresearch/mae) repo. Installation and preparation follow that repo.
* This repo is based on [timm](https://github.com/huggingface/pytorch-image-models) package, which is a PyTorch Image Models library.