# mosaic

This utility can be used to generate [photo-mosaic](http://en.wikipedia.org/wiki/Photographic_mosaic) images, to use it you must have Python installed, along with the [Pillow](http://pillow.readthedocs.org/en/latest/) imaging library.

As well as an image to use for the photo-mosaic ([most common image formats are supported](http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html)), you will need a large collection of different images to be used as tiles. The tile images can be any shape or size (the utility will automatically crop and resize them) but for good results you will need a lot of them - a few hundred at least. One convenient way of generating large numbers of tile images is to [extract screenshots from video files](http://linuxers.org/tutorial/how-extract-images-video-using-ffmpeg) using [ffmpeg](https://www.ffmpeg.org/).

Run the utility from the command line, as follows:

<pre>python mosaic.py <image> <tiles directory>
</pre>

*   The `image` argument should contain the path to the image for which you want to build the mosaic
*   The `tiles directory` argument should contain the path to the directory containing the tile images (the directory will be searched recursively, so it doesn't matter if some of the images are contained in sub-directories)

For example:

<pre>python mosaic.py game_of_thrones_poster.jpg /home/admin/images/screenshots
</pre>

The images below show an example of how the mosaic tiles are matched to the details of the original image:

![Mosaic Image](http://codebox.org.uk/graphics/mosaic/mosaic_small.jpg)  
<span class="smallText">Original</span>

[![Mosaic Image Detail](http://codebox.org.uk/graphics/mosaic/mosaic_detail.jpg)](http://codebox.org.uk/graphics/mosaic/mosaic_large.jpg)  
<span class="smallText">Mosaic Detail (click through for [full mosaic](http://codebox.org.uk/graphics/mosaic/mosaic_large.jpg) ~15MB)</span>

Producing large, highly detailed mosaics can take some time - you should experiment with the various [configuration parameters](https://github.com/codebox/mosaic/blob/master/mosaic.py#L6) explained in the source code to find the right balance between image quality and render time.

In particular the [TILE_MATCH_RES](https://github.com/codebox/mosaic/blob/master/mosaic.py#L8) parameter can have a big impact on both these factors - its value determines how closely the program examines each tile when trying to find the best fit for a particular segment of the image. Setting TILE_MATCH_RES to '1' simply finds the average colour of each tile, and picks the one that most closely matches the average colour of the image segment. As the value is increased, the tile is examined in more detail. Setting TILE_MATCH_RES to equal TILE_SIZE will cause the utility to examine each pixel in the tile individually, producing the best possible match (during my testing I didn't find a very noticeable improvement beyond a value of 5, but YMMV).

By default the utility will configure itself to use all available CPUs/CPU-cores on the host system, if you want to leave some processing power spare for other tasks then adjust the [WORKER_COUNT](https://github.com/codebox/mosaic/blob/master/mosaic.py#L12) parameter accordingly.
