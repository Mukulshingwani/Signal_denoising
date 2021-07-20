
# EEL2010 Programming Assignment

This folder contains various files which provide various building blocks, these blocks are finally assembled in file `main.py` and `ops.py` to solve the problem statement.

This README is divided into following sections:

- [Getting Started](#getting-started)

    - [Installing The Dependencies](#installing-the-dependencies)
    - [Testing Our Code](#testing-our-code)
        - [Why and How are the these tests performed](#why-and-how-are-these-tests-performed)

## Getting Started
We outline below the steps required to test and run our code.

### Installing the dependencies
To install all the required packages, simply switch to this directory (the directory containing this README file) in terminal
and run the following command (please make sure that `pip` is up-to-date)
```
pip install -r requirements.txt
```
This will install/update the following libraries:

- `NumPy`: Required for fast numerical computation on arrays.

- `matplotlib`: Required for plotting the signals.

- `pytest`: Required in order to test the building blocks of our code with just a single command.


### Testing our code
Once all the dependencies are installed, testing our code is as easy as running the following command in the terminal.
```
pytest .
```
This command will test all the functions which are fundamental to this programming assignment.
To be precise, following functions are tested (each of which is implemented by us):
- `conv1d`: convolution function used to convolve two signals

- `zero_pad`: this function is used to pad the signal with zeros. This function exists because we need to pad the signal
   before convolution so that the length of signal before and after convolution is same.

- `discrete_fourier_transform`: evident from the name itself, this function function calculates the Discrete Fourier Transform
   of a given signal.

- `inverse_fourier_transform`: calculates the Inverse Fourier Transform of a given Discrete Fourier Transform.

P.S.: All of these functions are defined in `utils/ops_utils.py`. More explanation about these functions is waiting in the next section(s).
And all the functions to test these functions are stored in `utils/test_ops_utils.py`


### Why and How are these tests performed ??
All the functions that we define in `utils/ops_utils.py` are used to further build the functions for
Denoising (`denoise`) and DeBlurring (`deblur`) in the file `ops.py` and then we use `deblur` and `denoise`
functions in `main.py` to actually solve the problem statement.

So it is evident that the functions defined in `utils/ops_utils.py` are the stepping stones for us. These are
the base functions on which everything is dependent for solving this problem.
And thus it is very important to test the implementation of these functions so that we can be assured that the
final results are accurate.

We test our implementation of these functions by comparing the results from our implementation with their `NumPy`
counterparts. For example, here is how we tested our implementation of `conv1d` function: We take two signals `a` and `b`,
then we convolve them using our `conv1d` functions and then we check if the result produced by our function matches with the result
that we get when we convolve `a` and `b` using `NumPy`'s `np.convolve` function.

P.S.: all of the functions to test our implementation are available in the file `utils/test_ops_utils.py`.