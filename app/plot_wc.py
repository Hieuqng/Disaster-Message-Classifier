import base64
from io import BytesIO

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from PIL import Image


def get_wordcloud(text, mask_image=None):
    stopwords = set(STOPWORDS)
    stopwords.update(["said", "will", "well", "one", "Thank", "Sandy"])

    if mask_image is not None:
        def transform_format(val):
            """
            Requires all white part of the mask should be 255 not 0
            """
            if val < 100:
                return 255
            else:
                return 0

        mask = np.array(Image.open(mask_image))
        squeeze_mask = mask.max(axis=2)
        # Transform your mask into a new one that will work with the function:
        transformed_mask = np.ndarray((squeeze_mask.shape[0], squeeze_mask.shape[1]), np.int32)

        for i in range(len(squeeze_mask)):
            transformed_mask[i] = list(map(transform_format, squeeze_mask[i]))

        # Word Cloud object
        wc = WordCloud(background_color="white", max_words=1000, mask=transformed_mask,
                       stopwords=stopwords, contour_width=3, contour_color='darkslateblue')

    else:
        wc = WordCloud(background_color="white", stopwords=stopwords)

    # Generate a wordcloud
    wc.generate(text)

    # Render image
    image = wc.to_image()

    return image


def return_image(df):
    messages = df['message']
    text = " ".join(review for review in messages.values)

    image = get_wordcloud(text, mask_image='./static/images/mask.png')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')
