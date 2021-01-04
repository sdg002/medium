# Introduction
In this article I would like to discuss some nuances about the `DBSCAN` clustering algorithm. This is by no means a detailed explanation of the algorithm. The intent of this article is to demonstrate how to calculate the value of `epsilon` parameter when the various dimensions of the features follow different measurement systems. E.g. Age on X axis and Height on Y axis.

# An overview of the DBSCAN algorithm
to be done
**Picture comes here**

# In depth videos to undersand DBSCAN
https://www.coursera.org/lecture/machine-learning-with-python/dbscan-B8ctK?utm_source=link&utm_medium=page_share&utm_content=vlp&utm_campaign=top_button

<iframe width="560" height="315" src="https://www.youtube.com/embed/5cOhL4B5waU" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


# Very simple code snippet to demonstrate DBSCAN
**Show matplotlib output here**

# Sample python code using Scikit image
**Python code snippet comes here**

```
import os
from skimage import io
from skimage.filters.rank import median
from skimage.morphology import disk


def median_filter(inputfilename):
    """
    Produces a blank image
    """
    folder_script=os.path.dirname(__file__)
    absolute_filename=os.path.join(folder_script,"./in/",inputfilename)

    original = io.imread(absolute_filename, as_gray=True)
    print(original.dtype)
    print(original.shape)
    print(original.max())
    print(original.min())
    
    median_filtered=median(original, disk(50))  #5, 10, 20,100

    filename_result="median-filter-output.png"
    file_result=os.path.join(folder_script,"./out/",filename_result)
    io.imsave(file_result,median_filtered)

median_filter(inputfilename="Sine-50-percent.png")

```

# Demo Gist
<script src="https://gist.github.com/sdg002/b825b141868e0dc33ec739e0b9a574f8.js"></script>
this does not work!!


# What happens when the dimensions of a feature have different units of measurements?
To be done
Present the problem here. A toy dataset of age, weight or age, height


# Conclusion
to be done


# References
- https://en.wikipedia.org/wiki/DBSCAN
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- 