# Beating-CAPTCHAs
### Beating CAPTCHAs with Neural Networks - project from Learning Data Mining with Python - Second Edition by Robert Layton

Images pose interesting and difficult challenges for data miners. Until recently, only small amounts of progress were made with analyzing images for extracting information. However recently, such as with the progress made on self-driving cars, significant advances have been made in a very short time-frame. The latest research is providing algorithms that can understand images for commercial surveillance, self-driving vehicles, and person identification.

There is lots of raw data in an image, and the standard method for encoding images - pixels - isn't that informative by itself. Images and photos can be blurry, too close to the targets, too dark, too light, scaled, cropped, skewed, or any other of a variety of problems that cause havoc for a computer system trying to extract useful information. Neural networks can combine these lower level features into higher level patterns that are more able to generalize and deal with these issues.

We are extracting text data from images by using neural networks for predicting each letter in the CAPTCHA. CAPTCHAs are images designed to be easy for humans to solve and hard for a computer to solve, as per the acronym: Completely Automated Public Turing test to tell Computers and Humans Apart. Many websites use them for registration and commenting systems to stop automated programs flooding their site with fake accounts and spam comments.

These tests help stop programs (bots) using websites, such as a bot intent on automatically signing up new people to a website. We play the part of such a spammer, trying to get around a CAPTCHA-protected system for posting messages to an online forum. The website is protected by a CAPTCHA, meaning we can't post unless we pass the test.

The topics covered in this projects are:

* Neural networks
* Creating our own dataset of CAPTCHAs and letters
* The scikit-image library for working with image data
* Extracting basic features from images
* Using neural networks for larger-scale classification tasks
* Improving performance using postprocessing
* Artificial neural networks

## Creating the dataset
In this chapter, to spice up things a little, let us take on the role of the bad guy. We want to create a program that can beat CAPTCHAs, allowing our comment spam program to advertise on someone's website. It should be noted that our CAPTCHAs will be a little easier than those used on the web today and that spamming isn't a very nice thing to do.

> We play the bad guy today, but please don't use this against real world sites. One reason to "play the bad guy" is to help improve the security of our website, by looking for issues with it.

Our experiment will simplify a CAPTCHA to be individual English words of four letters only, as shown in the following image:

Our goal will be to create a program that can recover the word from images like this. To do this, we will use four steps:

1. Break the image into individual letters.
2. Classify each individual letter.
3. Recombine the letters to form a word.
4. Rank words with a dictionary to try to fix errors.

> Our CAPTCHA-busting algorithm will make the following assumptions. First, the word will be a whole and valid four-character English word (in fact, we use the same dictionary for creating and busting CAPTCHAs). Second, the word will only contain uppercase letters. No symbols, numbers, or spaces will be used.

We are going to make the problem slightly harder than simply identifying letters, by performing a shear transform to the text, along with varying rates of shearing and scaling.

## Drawing basic CAPTCHAs
Before we can start classifying CAPTCHAs, we first need a dataset to learn from. In this section, we will be generating our own data to perform the data mining on.

> In more real-world applications, you'll be wanting to use an existing CAPTCHA service to generate the data, but for our purposes in this chapter, our own data will be sufficient. One of the issues that can arise is that we code in our assumptions around how the data works when we create the dataset ourselves, and then carry those same assumptions over to our data mining training.

Our goal here is to draw an image with a word on it, along with a shear transform. We are going to use the PIL library to draw our CAPTCHAs and the scikit-image library to perform the shear transform. The scikit-image library can read images in a NumPy array format that PIL can export to, allowing us to use both libraries.

> Both PIL and scikit-image can be installed via Anaconda. However, I recommend getting PIL through its replacement called pillow:
> conda install pillow scikit-image

First, we import the necessary libraries and modules. We import NumPy and the Image drawing functions.

Then we create our base function for generating CAPTCHAs. This function takes a word and a shear value (which is normally between 0 and 0.5) to return an image in a NumPy array format. We allow the user to set the size of the resulting image, as we will use this function for single-letter training samples.

In this function, we create a new image using L for the format, which means black-and-white pixels only, and create an instance of the ImageDraw class. This allows us to draw on this image using PIL. We then load the font, draw the text, and perform a scikit-image shear transform on it. 

> You can get the Coval font I used from the Open Font Library at:
> [http://openfontlibrary.org/en/font/bretan](http://openfontlibrary.org/en/font/bretan)
> Download the .zip file and extract the Coval-Black.otf file into the same directory as your Notebook.

From here, we can now generate images quite easily and use pyplot to display them. First, we use our inline display for the matplotlib graphs and import pyplot.

## Splitting the image into individual letters
Our CAPTCHAs are words. Instead of building a classifier that can identify the thousands and thousands of possible words, we will break the problem down into a smaller problem: predicting letters.

> Our experiment is in English, and all uppercase, meaning we have 26 classes to predict from for each letter. If you try these experiments in other languages, keep in mind the number of output classes will have to change.
The first step in our algorithm for beating these CAPTCHAs involves segmenting the word to discover each of the letters within it. To do this, we are going to create a function that finds contiguous sections of black pixels in the image and extract them as subimages. These are (or at least should be) our letters. The scikit-image function has tools for performing these operations.

Our function will take an image, and return a list of sub-images, where each sub-image is a letter from the original word in the image. The first thing we need to do is to detect where each letter is. To do this, we will use the label function in scikit-image, which finds connected sets of pixels that have the same value.

We will then extract the subimages from the example CAPTCHA.

As you can see, our image segmentation does a reasonable job, but the results are still quite messy, with bits of previous letters showing. This is fine, and almost preferable. While training on data with regular noise makes our training worse, training on data with random noise can actually make it better. One reason is that the underlying data mining model learns the important aspects, namely the non-noise parts instead of specific noise inherent in the training data set. It is a fine line between too much and too little noise, and this can be hard to properly model. Testing on validation sets is a good way to ensure your training is improving.

One important note is that this code is not consistent in finding letters. Lower shear values typically result in accurately segmented images.

## Creating a training dataset
Using the functions we have already defined, we can now create a dataset of letters, each with different shear values. From this, we will train a neural network to recognize each letter from the image.

We first set up our random state and an array that holds the options for letters, shear values and scale values that we will randomly select from. There isn't much surprise here, but if you haven't used NumPy's arange function before, it is similar to Python's range function—except this one works with NumPy arrays and allows the step to be a float.

We then create a function (for generating a single sample in our training dataset) that randomly selects a letter, a shear value, and a scale value selected from the available options.

We return the image of the letter, along with the target value representing the letter in the image. Our classes will be 0 for A, 1 for B, 2 for C, and so on.

The resulting image has just a single letter, with a random shear and random scale value.

We can now generate all of our data by calling this several thousand times. We then put the data into NumPy arrays, as they are easier to work with than lists.

Our targets are integer values between 0 and 26, with each representing a letter of the alphabet. Neural networks don't usually support multiple values from a single neuron, instead preferring to have multiple outputs, each with values 0 or 1. We perform one-hot-encoding of the targets, giving us a target array that has 26 outputs per sample, using values near 1 if that letter is likely and near 0 otherwise.

From this output, we know that our neural network's output layer will have 26 neurons. The goal of the neural network is to determine which of these neurons to fire, based on a given input--the pixels that compose the image.

The library we are going to use doesn't support sparse arrays, so we need to turn our sparse matrix into a dense NumPy array.

Finally, we perform a train/test split to later evaluate our data.

## Training and classifying
We are now going to build a neural network that will take an image as input and try to predict which (single) letter is in the image.

We will use the training set of single letters we created earlier. The dataset itself is quite simple. We have a 20-by-20-pixel image, each pixel 1 (black) or 0 (white). These represent the 400 features that we will use as inputs into the neural network. The outputs will be 26 values between 0 and 1, where higher values indicate a higher likelihood that the associated letter (the first neuron is A, the second is B, and so on) is the letter represented by the input image.

We are going to use the scikit-learn's MLPClassifier for our neural network in this chapter.

> You will need a recent version of scikit-learn to use MLPClassifier. If the below import statement fails, try again after updating scikit-learn. You can do this using the following Anaconda command:
 conda update scikit-learn
As for other scikit-learn classifiers, we import the model type and create a new one. The constructor below specifies that we create one hidden layer with 100 nodes in it.

To see the internal parameters of the neural network, we can use the get_params() function. This function exists on all scikit-learn models. Here is the output from the above model. Many of these parameters can improve training or the speed of training.

Next, we fit our model using the standard scikit-learn interface.

Our model has now learned weights between each of the layers. We can view those weights by examining clf.coefs_, which is a list of NumPy arrays that join each of the layers. For example, the weights between the input layer with 400 neurons (from each of our pixels) to the hidden layer with 100 neurons (a parameter we set), can be obtained using clf.coefs_[0]. In addition, the weights between the hidden layer and the output layer (with 26 neurons) can be obtained using clf.coefs_[1]. These weights, together with the parameters above, wholly define our trained network.

We can now use that trained network to predict our test dataset.

The result is 0.96, which is pretty impressive. This version of the F1 score is based on the macro-average, which computes the individual F1 score for each class, and then averages them without considering the size of each class.

The final f1-score for this report is shown on the bottom right, the second last number - 0.99. This is the micro-average, where the f1-score is computed for each sample and then the mean is computed. This form makes more sense for relatively similar class sizes, while the macro-average makes more sense for imbalanced classes.

Pretty simple from an API perspective, as scikit-learn hides all of the complexity. However what actually happened in the backend? How do we train a neural network?

## Back-propagation
Training a neural network is specifically focused on the following things.

* The first is the size and shape of the network - how many layers, what sized layers and what error functions they use. While types of neural networks exists that can alter their size and shape, the most common type, a feed-forward neural network, rarely has this capability. Instead, its size is fixed at initialization time, which in this chapter is 400 neurons in the first layer, 100 in the hidden layer and 26 in the final layer. Training for the shape is usually the job of a meta-algorithm that trains a set of neural networks and determines which is the most effective, outside of training the networks themselves.
* The second part of training a neural network is to alter the weights between neurons. In a standard neural network, nodes from one layer are attached to nodes of the next layer by edges with a specific weight. These can be initialized randomly (although several smarter methods do exist such as autoencoders), but need to be adjusted to allow the network to learn the relationship between training samples and training classes.

This adjusting of weights was one of the key issues holding back very-early neural networks, before an algorithm called back propagation was developed to solve the issue.
