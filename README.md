# Labels2Rois
## 1. Overview
This is an OMERO.web script to convert grey scale label images into OMERO ROIs.<br>
Depending on the python packages installed in the OMERO server virtual environment one can choose between Poylgon or Mask as
a Shape for the ROIs.<br>

## 2. Naming convention
For now the script relies on the label images having the same exact name as the original image with "-label" added to the end.<br>
For example `Larvae_3w_400l_trRL.ome.tiff` will need a label image `Larvae_3w_400l_trRL-label.ome.tif` for the script to recognize it.<br>
This label image needs to be in the same dataset as the its "original" image.



## 3. Package dependencies
The script tries to import [omero_rois](https://github.com/ome/omero-rois/) and [scikit-image](https://github.com/scikit-image/scikit-image) and will give you feedback if those are not installed. Please consult your OMERO system-admin for installation.<br>
The reason two packages are needed, is to give you the option what type of ``omero.model.Shape`` you want to have for your ROI.<br>
>What is a **Shape**?:<br>
An OMERO ROI (`omero.model.Roi`) is a container object consisting of one or multiple Shapes (e.g. a Line, Polyline, Polygon, Mask, etc.) which constitute the actual forms/shapes you see in the OMERO.iviewer.

If you want to avoid additional packages and the related dependency-bloat you can still create Polygon Shapes as is, with just a small increase in script runtime.<br>
To achieve that I refactored the underlying Cython code from the relevant `scikit-image` function into "pure" Python, therefore having only `numpy` as dependency.



### 3.1 Mask Shape
Creating Mask Shapes for the ROIs relies on the package `omero_rois` created by the OME team.

### 3.2 Polygon Shape
Creating Polygon Shapes for the ROIs relies on the `find_contours()` function from [scikit-image](https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py).<br>
In short this functions relies on the "marching squares" algorithm. For more details read the comments in the source code linked above.<br>
To install `scikit-image` on our OMERO instance we had to install the following previously uninstalled packages:
```
PyWavelets, cycler, decorator, imageion, kiwisolver, matplotlib, networkx, scipy, tifffile
```

<br>

Here is the dependency tree (made with `pipdeptree`) for the `scikit-image` installation we performed on our server (**green**: existing packages, **blue**: newly installed packages):

<img src="dependencies.svg" height="350">

## 3. Caveats
- At the moment the input Data type is limited to Datasets and Images.<br>
This can easily expanded in the future if the need arises, just contact the author via mail or image.sc.<br>
- The script has not been tested on really complex ROI patterns. There might be situations where the underlying `find_contours()` function from `scikit-image` will fail to produce an accurate Polygon ROI.<br>
The underlying function to create Mask ROIs though is independent of shape complexity and might provide a good fallback option in this case.

## 4. Outlook
If you use the script and see room for improvement or have a special use case that is not covered by the generic code I wrote, please write an issue here at Github or contact me via mail or [Image.sc](https://forum.image.sc/).<br>
I might implement some logic to artificially create a Polygon Shape from the Mask Shape the is created with `omero_rois` to better deal with complex ROI forms.<br>
To make it more generic, there might also be the option to put in a regex pattern to determine the label images from the selection.

