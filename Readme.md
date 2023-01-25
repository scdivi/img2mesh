img_2_nutils
============

This reposity provides a tool to convert .png to [Nutils][https://nutils.org/install-nutils.html] readable topology data structure.

Instructions for usage
----------------------
- Name you image as <image_name>_<pixel_size>.png 
- Save it in subfolder images/
- Create a .raw and .json wrapper from a png file
    - bash img2json.sh image_name pixel_size (for example bash img2json.sh walle 70)
- Go to examples subfolder and run example_2d script
    - python3 example_2d.py fname=<image_name>_<image_size> (for example python3 example_2d.py fname=walle_70)

Required (external) packages
----------------------------
- [Nutils][https://nutils.org/install-nutils.html]
- [Scikit-mage][https://scikit-image.org/docs/stable/install.html]