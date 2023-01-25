# img_2_nutils
This reposity provides a tool to convert .png to [Nutils](https://github.com/evalf/nutils) readable topology data structure (i.e. analysis-suitable geometry).

## Instructions for usage
- Name you image as `<image_name>_<pixel_size>.png` 
- Save it in subfolder images/
- Create a .raw and .json wrapper from a png file
    - `bash img2json.sh <image_name>_<pixel_size>` (for example `bash img2json.sh walle 35`)
- Go to examples subfolder and run example_2d script
    - `python3 example_2d.py fname=<image_name>_<image_size>` (for example `python3 example_2d.py fname=walle_35`)

## Topology-preservation
The tool contains a feature called topology preservation -- an alogrithm based on the [paper](https://doi.org/10.1016/j.cma.2022.114648). It can be activated with `topopreserve=True`. For example `python3 example_2d.py fname=walle_35 topopreserve=True`. Note that the sharpness of the reconstructed boundariese can be varied with `maxrefine` argument. For example `python3 example_2d.py fname=walle_35 maxrefine=3`.

### Before topology preservation:
![image](https://user-images.githubusercontent.com/33148729/214567864-4230b06a-630f-4255-a405-32612ea4c553.png)
### After topology preservation:
![image](https://user-images.githubusercontent.com/33148729/214567723-c28b9b64-ae5e-4310-a048-59d10c26a957.png)

## Example images:
### Pacman 300px X 300px
![pacman_300](https://user-images.githubusercontent.com/33148729/214568005-0beb9cb5-b2d5-44e8-b99e-56593f50e16a.png)
### Airfoil 100px X 100px
![airfoil_100](https://user-images.githubusercontent.com/33148729/214570382-caea3272-079a-4ab5-ae68-51c40c76ef18.png)
### Rocket 100px X 100px
![rocket_100](https://user-images.githubusercontent.com/33148729/214568017-eedd286f-6fa0-494f-910a-5161d52b8b11.png)
### WALL-E 70px X 70px
![walle_70](https://user-images.githubusercontent.com/33148729/214568039-c203cd85-f17f-498a-a8e7-b717704bd88c.png)
### Carotid aortary 32px X 32 px
![carotid_32](https://user-images.githubusercontent.com/33148729/214570436-5ff1c587-4e65-4afd-8b3b-535b572a03f6.png)

## Required (external) packages
- [Nutils](https://github.com/evalf/nutils)
- [Scikit-mage](https://github.com/scikit-image/scikit-image)
