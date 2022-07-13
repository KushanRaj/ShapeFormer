# ShapeFormer: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer

[Mukund Varma T](), [Kushan Raj](), [ Dimple A Shajahan](), [M. Ramanathan](), 


-----
This repository contains the source code for the paper [ShapeFormer: A Transformer for Point Cloud Completion]() (3DV 2022). The project page is [here](https://kushanraj.github.io/ShapeFormer/).



## Cite this work

```
```

[<img src="pics/network.png" width="100%" alt="Intro pic" />](pics/completion3d.png)

## Datasets

We use the [MVP](https://mvp-dataset.github.io/) and [Compeletion3D](http://completion3d.stanford.edu/) datasets in our experiments, which are available below:

- [MVP](https://mvp-dataset.github.io/MVP/Completion.html)
- [Completion3D](https://completion3d.stanford.edu/)

The pretrained models on Completion3D and PCN dataset are available as follows:

- [ShapeFormer_pre-trained](https://drive.google.com/drive/folders/1oO7HKsyQuOr6n4HOxe07yHjxchYDUGM-?usp=sharing)


#### Install Python Denpendencies

```
cd ShapeFormer
pip install -r requirements.txt
```

#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4 of cuda version are required.

```
cd pointnet2_ops_lib
python setup.py install

cd ..

cd Chamfer3D
python setup.py install

cd ..

cd emd
python setup.py install
```


## Acknowledgements

Some of the code of this repo is borrowed from [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch). We thank the authors for their great job!
