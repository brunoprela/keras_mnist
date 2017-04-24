# keras_mnist
Simple digit recognition in Keras using the MNIST data set. Just a quick intro I did a while ago to get acquainted with Keras. To use this project you will need Python3. Simply clone this repo, 
create and enter a **virtualenv**,* run '**pip install -r requirements.txt**', and then run '**python keras.mnist.py**'.

Helpful note if you are using **Python 3.6** on OSX so you don't tear your hair out. You might run into an error saying something along the lines of:
``` SSL: CERTIFICATE_VERIFY_FAILED ```
when trying to connect to an https:// site. This is because Python 3.6 on OSX does not have certificates and can't validate any SSL connections.
But you can remedy this problem with a post-install step which installs the certifi package of certificates. You can find documentation for this
in the README.md in the Python 3.6 folder in your Applications directory. Essentialy you'll need to run the file at **/Applications/Python\ 3.6/Install\ Certificates.command** 
and your problem should be fixed.
