# I2I-GANs

Common generative adversarial networks (GANs) implemented in TensorFlow 2.4.1. The GANs are suitable
for image-to-image translation tasks.

The repository was published as a part of the master's thesis (Generative Adversarial Networks Applied
for Privacy Preservation in Biometric-Based Authentication and Identification). Preliminary
results were presented at http://excel.fit.vutbr.cz/submissions/2021/031/31.pdf.

The following architectures are implemented:
- TraVeLGAN (https://github.com/KrishnaswamyLab/travelgan)
- DiscoGAN (https://github.com/SKTBrain/DiscoGAN)
- GcGAN (https://github.com/hufu6371/GcGAN)

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/lubosmj/I2I-GANs && cd I2I-GANs
   ```
2. Create a new virtual environment:
   ```
   python3 -m venv venv
   source source venv/bin/activate
   ```
3. Install the packages:
   ```
   python3 setup.py install
   ```
4. Use the installed modules in your application:
   ```python3
   from i2i_gans import TraVeLGAN
   
   travelgan = TraVeLGAN(...)
   travelgan.compile()
   travelgan.load_weights(...)
   
   fake_images = travelgan.generator(...)
   ```

### Running the Examples
1. Train a new TraVeLGAN model:
   ```
   python3 -m examples.travelgan_trainer train --domain_A "path/to/dataset/A/*.png" --domain_B "path/to/dataset/B/*.png" --dataset_size 5000 --batch_size=16 --checkpoints_freq 10 --parallel --samples_freq 10 --samples_dir samples --checkpoints_dir checkpoints --augment random_flip_left_right --epochs 250
   ```
2. Train a new DiscoGAN model:
   ```
   python3 -m examples.discogan_trainer train --domain_A "path/to/dataset/A/*.png" --domain_B "path/to/dataset/B/*.png" --dataset_size 5000 --batch_size=200 --checkpoints_freq 10 --parallel --samples_freq 10 --samples_dir samples --checkpoints_dir checkpoints --augment random_flip_left_right --epochs 200
   ```
3. Train a new GcGAN model:
   ```
   python3 -m examples.gcgan_trainer train --domain_A "path/to/dataset/A/*.png" --domain_B "path/to/dataset/B/*.png" --dataset_size 5000 --batch_size=12 --checkpoints_freq 10 --parallel --samples_freq 10 --samples_dir samples --checkpoints_dir checkpoints --augment random_flip_left_right --epochs 200
   ```

### Generated Images
#### TraVeLGAN
The GAN was trained for 250 epochs with Adam optimizer (learning rate: 0.0002, batch size: 16, dataset size: 8,000).
- Datasets:
  - Augmented images from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

<img src="https://user-images.githubusercontent.com/8740962/117123304-28234880-ad97-11eb-800e-35547f05d528.png" width="60%">

#### DiscoGAN
The GAN was trained for 200 epochs with the same hyper-parameters as recommended in the original paper (dataset size: 20,000). Additionally, one convolution layer with 100 filters was inserted into the generators.
- Datasets:
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [UT Zappos50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/)

<img src="https://user-images.githubusercontent.com/8740962/117123568-80f2e100-ad97-11eb-8bd1-47f42b8a3c1b.png" width="60%">
