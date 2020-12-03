# Neutralizer - Thesis Project

## Structure:

### data:

`./data/row` folder contains original datasets `jaffe` and `tfeid`

As we need more domains for DG, `kdef` dataset (european faces) is to be added.

### modules

`modules/data` package contains all the code needed to work with data. 
We have organized data in a torch `Dataset` and pytorch-lightning `DataModule`.

Each sample is a dictionary of the form:

```
{ 
  'image' : <torch.Tensor>,
  'desc'  : {
              'path': 'data/raw/jaffe/KA.AN1.39.tiff',
              'exp' : < 0-7 >, # expression class
              'iden': 'KA'     # identiy code
            }
}
```

If needed, optional `neutral` key is present in the data sample dictionary 
and contains a corresponding neutral sample list. For example :

```
{ 
  'image'   : <torch.Tensor>,
  'desc'    : {
                'path': 'data/raw/jaffe/KA.AN1.39.tiff',
                'exp' : < 0-7 >, # expression class
                'iden': 'KA'     # identiy code
              },
  'neutral' : [
                { 
                  'image' : <torch.Tensor>,
                  'desc'  : {
                              'path': 'data/raw/jaffe/KA.NE1.155.tiff',
                              'exp' : 0, # expression class
                              'iden': 'KA'     # identiy code
                            }
                },
                { 
                  'image' : <torch.Tensor>,
                  'desc'  : {
                              'path': 'data/raw/jaffe/KA.NE2.156.tiff',
                              'exp' : 0, # expression class
                              'iden': 'KA'     # identiy code
                            }
                },
                { 
                  'image' : <torch.Tensor>,
                  'desc'  : {
                              'path': 'data/raw/jaffe/KA.NE3.157.tiff',
                              'exp' : 0, # expression class
                              'iden': 'KA'     # identiy code
                            }
                }
              ]
}
```


DataModules support 3 scenarios: 
- `exp`   - classical train-test split stratified by expression classes.
- `train` - for training on the whole dataset 
- `test`  - for testing on the whole dataset

As one of the transforms used is Normalization with `mean = 0.5`, `std = 0.5` 
which basically means that image is mapped in the range of `[-1, 1]` for 
better performance of models, so it needs to be denormalize back when viewed.

For this I have made `modules.data.trnsforms.DeNormalize` transform class.

### notebooks

This section just contains some notebooks, that were not moved to the project structure.

### vanilla

As I separate `LightningModule` classes from `torch.nn` classes, `vanilla` models are placed here.

Vanilla modules are basically components, that allow us to construct more complex models as neutralizer, etc.

Here we have: 
- Deep Emotion - sota solution to test with
- VGG-16 classifier, adapted (

should I add a 1->3 convolutional layer or substitute the first one from the pretrained model?

)
- VGG-16 based Encoder (pretrained on ImageNet)
- Decoder
- InterpolateUp layer for upsampling
- ReverseGradLayer - layer that reverses gradients (multiplies by -1 and some constant)